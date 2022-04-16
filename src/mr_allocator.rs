use crate::{
    memory_region::{local::LocalMr, RawMemoryRegion},
    protection_domain::ProtectionDomain,
    LocalMrReadAccess,
};
use clippy_utilities::OverflowArithmetic;
use libc::{c_void, size_t};
use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
use num_traits::ToPrimitive;
use rdma_sys::ibv_access_flags;
use std::collections::BTreeMap;
use std::mem;
use std::sync::Mutex;
use std::{alloc::Layout, io, ptr, sync::Arc};
use tikv_jemalloc_sys::{self, extent_hooks_t, MALLOCX_ALIGN, MALLOCX_ARENA};
use tracing::{debug, error};

/// Get default extent hooks from arena0
static DEFAULT_ARENA_INDEX: u32 = 0;

/// Custom extent alloc hook used by jemalloc
/// Alloc extent memory with `ibv_reg_mr()`
static RDMA_ALLOC_EXTENT_HOOK: unsafe extern "C" fn(
    extent_hooks: *mut extent_hooks_t,
    new_addr: *mut c_void,
    size: usize,
    alignment: usize,
    zero: *mut i32,
    commit: *mut i32,
    arena_ind: u32,
) -> *mut c_void = extent_alloc_hook;

/// Custom extent dalloc hook used by jemalloc
/// Dalloc extent memory with `ibv_dereg_mr()`
static RDMA_DALLOC_EXTENT_HOOK: unsafe extern "C" fn(
    extent_hooks: *mut extent_hooks_t,
    addr: *mut c_void,
    size: usize,
    committed: i32,
    arena_ind: u32,
) -> i32 = extent_dalloc_hook;

/// Custom extent merge hook used by jemalloc
/// Merge two adjacent extents to a bigger one
static RDMA_MERGE_EXTENT_HOOK: unsafe extern "C" fn(
    extent_hooks: *mut extent_hooks_t,
    addr_a: *mut c_void,
    size_a: usize,
    addr_b: *mut c_void,
    size_b: usize,
    committed: i32,
    arena_ind: u32,
) -> i32 = extent_merge_hook;

/// Custom extent hooks
static mut RDMA_EXTENT_HOOKS: extent_hooks_t = extent_hooks_t {
    alloc: Some(RDMA_ALLOC_EXTENT_HOOK),
    dalloc: Some(RDMA_DALLOC_EXTENT_HOOK),
    destroy: None,
    commit: None,
    decommit: None,
    purge_lazy: None,
    purge_forced: None,
    split: None,
    merge: Some(RDMA_MERGE_EXTENT_HOOK),
};

lazy_static! {
    /// Default extent hooks of jemalloc
    static ref ORIGIN_HOOKS: extent_hooks_t = get_default_hooks_impl(DEFAULT_ARENA_INDEX).unwrap();
    /// The correspondence between extent metadata and `raw_mr`
    #[derive(Debug)]
    pub(crate) static ref EXTENT_TOKEN_MAP: Arc<Mutex<BTreeMap<usize, Item>>> = Arc::new(Mutex::new(BTreeMap::<usize, Item>::new()));
    /// The correspondence between `arena_ind` and `ProtectionDomain`
    pub(crate) static ref ARENA_PD_MAP: Arc<LockFreeCuckooHash<u32, Arc<ProtectionDomain>>> = Arc::new(LockFreeCuckooHash::new());
}

/// Combination between extent metadata and `raw_mr`
#[derive(Debug)]
pub(crate) struct Item {
    addr: usize,
    len: usize,
    raw_mr: Arc<RawMemoryRegion>,
    _arena_ind: u32,
}

/// Memory region allocator
#[derive(Debug)]
pub(crate) struct MrAllocator {
    /// Protection domain that holds the allocator
    _pd: Arc<ProtectionDomain>,
    /// Arena index
    arena_ind: u32,
}

impl MrAllocator {
    /// Create a new MR allocator
    pub(crate) fn new(pd: Arc<ProtectionDomain>) -> Self {
        get_je_stats();
        let arena_ind = init_je_statics(pd.clone()).unwrap();
        Self { _pd: pd, arena_ind }
    }

    /// Allocate a MR according to the `layout`
    pub(crate) fn alloc(self: &Arc<Self>, layout: &Layout) -> io::Result<LocalMr> {
        let addr = self.alloc_from_je(layout) as usize;
        let raw_mr = self.lookup_raw_mr(addr).unwrap();
        Ok(LocalMr::new(addr, layout.size(), raw_mr))
    }

    /// Alloc memory for RDMA operations from jemalloc
    fn alloc_from_je(&self, layout: &Layout) -> *mut u8 {
        let addr = unsafe {
            tikv_jemalloc_sys::mallocx(
                layout.size(),
                (MALLOCX_ALIGN(layout.align()) | MALLOCX_ARENA(self.arena_ind.to_usize().unwrap()))
                    .to_i32()
                    .unwrap(),
            )
        };
        assert_ne!(addr, ptr::null_mut());
        addr as *mut u8
    }

    /// Look up `raw_mr` info by addr
    fn lookup_raw_mr(&self, addr: usize) -> Option<Arc<RawMemoryRegion>> {
        match EXTENT_TOKEN_MAP
            .lock()
            .unwrap()
            .range(..addr.overflow_add(1))
            .next_back()
        {
            Some((_, item)) => {
                debug!("LOOK addr {}, item {:?}", addr, item);
                assert_eq!(self.arena_ind, item._arena_ind);
                assert!(addr >= item.addr && addr < item.addr + item.len);
                Some(item.raw_mr.clone())
            }
            None => {
                error!("can not find raw mr by addr {}", addr);
                None
            }
        }
    }
}

/// Get default extent hooks of jemalloc
#[allow(trivial_casts)] // `cast() doesn't work here
fn get_default_hooks_impl(arena_ind: u32) -> io::Result<extent_hooks_t> {
    // read default alloc impl
    let mut hooks: *mut extent_hooks_t = ptr::null_mut();
    let key = format!("arena.{}.extent_hooks\0", arena_ind);
    mem::forget(hooks);
    let mut hooks_len = mem::size_of_val(&hooks);
    let errno = unsafe {
        tikv_jemalloc_sys::mallctl(
            key.as_ptr() as *const _,
            &mut hooks as *mut _ as *mut c_void,
            &mut hooks_len,
            ptr::null_mut(),
            0,
        )
    };

    let hooksd = unsafe { *hooks };
    if errno != 0_i32 {
        return Err(io::Error::from_raw_os_error(errno));
    }

    Ok(hooksd)
}

/// Set custom extent hooks
#[allow(trivial_casts)] // `cast() doesn't work here
fn set_extent_hooks(arena_ind: u32) -> io::Result<()> {
    let key = format!("arena.{}.extent_hooks\0", arena_ind);
    let hooks_len: size_t = unsafe { mem::size_of_val(&&RDMA_EXTENT_HOOKS) };
    let errno = unsafe {
        tikv_jemalloc_sys::mallctl(
            key.as_ptr() as *const _,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut &mut RDMA_EXTENT_HOOKS as *mut _ as *mut c_void,
            hooks_len,
        )
    };
    debug!("arena<{}> set hooks success", arena_ind);
    if errno != 0_i32 {
        error!("set extent hooks failed");
        return Err(io::Error::from_raw_os_error(errno));
    }
    Ok(())
}

/// Create an arena to manage `MR`s memory
#[allow(trivial_casts)] // `cast() doesn't work here
fn create_arena() -> io::Result<u32> {
    let key = "arenas.create\0";
    let mut aid = 0_u32;
    let mut aid_len: size_t = mem::size_of_val(&aid);
    let errno = unsafe {
        tikv_jemalloc_sys::mallctl(
            key.as_ptr() as *const _,
            &mut aid as *mut _ as *mut c_void,
            &mut aid_len,
            ptr::null_mut(),
            0,
        )
    };
    if errno != 0_i32 {
        error!("set extent hooks failed");
        return Err(io::Error::from_raw_os_error(errno));
    }
    debug!("create arena success aid : {}", aid);
    Ok(aid)
}

/// Create arena and init statics
fn init_je_statics(pd: Arc<ProtectionDomain>) -> io::Result<u32> {
    let ind = create_arena()?;
    if ARENA_PD_MAP.insert(ind, pd) {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "insert ARENA_PD_MAP failed",
        ));
    }
    set_extent_hooks(ind)?;
    Ok(ind)
}

/// Get stats of je
/// TODO: Need to optimize
#[allow(trivial_casts)] // `cast() doesn't work here
fn get_je_stats() {
    let mut allocated: usize = 0;
    let mut val_len = mem::size_of_val(&allocated);
    let field = "stats.allocated\0";
    let errno = unsafe {
        tikv_jemalloc_sys::mallctl(
            field.as_ptr() as *const _,
            &mut allocated as *mut _ as *mut c_void,
            &mut val_len,
            ptr::null_mut(),
            0,
        )
    };
    if errno != 0_i32 {
        error!("can not get je");
    }
    debug!("allocated {}", allocated);
}

unsafe extern "C" fn extent_alloc_hook(
    extent_hooks: *mut extent_hooks_t,
    new_addr: *mut c_void,
    size: usize,
    alignment: usize,
    zero: *mut i32,
    commit: *mut i32,
    arena_ind: u32,
) -> *mut c_void {
    let origin_alloc = (*ORIGIN_HOOKS).alloc.unwrap();
    let addr = origin_alloc(
        extent_hooks,
        new_addr,
        size,
        alignment,
        zero,
        commit,
        arena_ind,
    );
    let item = register_extent_mr_default(addr, size, arena_ind);
    debug!("ALLOC item {:?} lkey {}", &item, item.raw_mr.lkey());
    match EXTENT_TOKEN_MAP.lock().unwrap().insert(item.addr, item) {
        Some(_) => {
            panic!("alloc the same addr double time");
        }
        None => addr,
    }
}

unsafe extern "C" fn extent_dalloc_hook(
    extent_hooks: *mut extent_hooks_t,
    addr: *mut c_void,
    size: usize,
    committed: i32,
    arena_ind: u32,
) -> i32 {
    debug!("DALLOC addr {}, size{}", addr as usize, size);
    let _ = EXTENT_TOKEN_MAP
        .lock()
        .unwrap()
        .remove(&(addr as _))
        .unwrap();
    let origin_dalloc = (*ORIGIN_HOOKS).dalloc.unwrap();
    origin_dalloc(extent_hooks, addr, size, committed, arena_ind)
}

unsafe extern "C" fn extent_merge_hook(
    extent_hooks: *mut extent_hooks_t,
    addr_a: *mut c_void,
    size_a: usize,
    addr_b: *mut c_void,
    size_b: usize,
    committed: i32,
    arena_ind: u32,
) -> i32 {
    let origin_merge = (*ORIGIN_HOOKS).merge.unwrap();
    let err = origin_merge(
        extent_hooks,
        addr_a,
        size_a,
        addr_b,
        size_b,
        committed,
        arena_ind,
    );
    debug!(
        "MERGE addr_a {}, size_a {}; addr_b {}, size_b {} err {}",
        addr_a as usize, size_a, addr_b as usize, size_b, err
    );
    if err != 0 {
        return 1_i32;
    }
    let arena_a = EXTENT_TOKEN_MAP
        .lock()
        .unwrap()
        .get(&(addr_a as usize))
        .unwrap()
        ._arena_ind;
    let arena_b = EXTENT_TOKEN_MAP
        .lock()
        .unwrap()
        .get(&(addr_b as usize))
        .unwrap()
        ._arena_ind;
    // make sure the extents belong to the same pd(arena).
    if arena_a != arena_b {
        return 1_i32;
    }
    let _ = EXTENT_TOKEN_MAP
        .lock()
        .unwrap()
        .remove(&(addr_a as _))
        .unwrap();
    let _ = EXTENT_TOKEN_MAP
        .lock()
        .unwrap()
        .remove(&(addr_b as _))
        .unwrap();
    // the old mrs will deregister after `raw_mr` dorp
    // so we only need to register a new `raw_mr`
    let item = register_extent_mr_default(addr_a, size_a.overflow_add(size_b), arena_ind);
    match EXTENT_TOKEN_MAP.lock().unwrap().insert(item.addr, item) {
        Some(_) => {
            panic!("alloc the same addr double time");
        }
        None => 0_i32,
    }
}

/// Register extent memory region with default access flags
pub(crate) fn register_extent_mr_default(addr: *mut c_void, size: usize, arena_ind: u32) -> Item {
    let access = ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
        | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
        | ibv_access_flags::IBV_ACCESS_REMOTE_READ
        | ibv_access_flags::IBV_ACCESS_REMOTE_ATOMIC;
    register_extent_mr(addr, size, arena_ind, access)
}

/// Register extent memory region
pub(crate) fn register_extent_mr(
    addr: *mut c_void,
    size: usize,
    arena_ind: u32,
    access: ibv_access_flags,
) -> Item {
    assert_ne!(addr, ptr::null_mut());
    let raw_mr = Arc::new(
        RawMemoryRegion::register_from_pd(
            ARENA_PD_MAP.get(&arena_ind, &pin()).unwrap(),
            addr as *mut u8,
            size,
            access,
        )
        .unwrap(),
    );
    Item {
        addr: addr as usize,
        len: size,
        raw_mr,
        _arena_ind: arena_ind,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::Context, RdmaBuilder};
    use std::{alloc::Layout, io, thread};
    use tikv_jemalloc_sys::MALLOCX_ALIGN;

    #[tokio::test]
    async fn alloc_mr_from_rdma() -> io::Result<()> {
        let rdma = RdmaBuilder::default()
            .set_port_num(1)
            .set_gid_index(1)
            .build()?;
        let mut mrs = vec![];
        //Layout::new::<[u8; 4096]>()
        let layout = Layout::new::<char>();
        for _ in 0_i32..2_i32 {
            let mr = rdma.alloc_local_mr(layout)?;
            mrs.push(mr);
        }
        Ok(())
    }

    #[test]
    fn alloc_mr_from_allocator() -> io::Result<()> {
        let ctx = Arc::new(Context::open(None, 1, 1)?);
        let pd = Arc::new(ctx.create_protection_domain()?);
        let allocator = Arc::new(MrAllocator::new(pd));
        let layout = Layout::new::<char>();
        let lmr = allocator.alloc(&layout)?;
        debug!("lmr info :{:?}", &lmr);
        Ok(())
    }

    #[test]
    fn je_malloxc_test() {
        let ctx = Arc::new(Context::open(None, 1, 1).unwrap());
        let pd = Arc::new(ctx.create_protection_domain().unwrap());
        let ind = init_je_statics(pd.clone()).unwrap();
        let thread = thread::spawn(move || {
            let layout = Layout::new::<char>();
            let addr = unsafe {
                tikv_jemalloc_sys::mallocx(
                    layout.size(),
                    MALLOCX_ALIGN(layout.align()) | MALLOCX_ARENA(ind.to_usize().unwrap()),
                )
            };
            assert_ne!(addr, ptr::null_mut());
            unsafe {
                *(addr as *mut char) = 'c';
                assert_eq!(*(addr as *mut char), 'c');
                debug!("addr : {}, char : {}", addr as usize, *(addr as *mut char));
            }
        });
        let layout = Layout::new::<char>();
        let addr = unsafe {
            tikv_jemalloc_sys::mallocx(
                layout.size(),
                MALLOCX_ALIGN(layout.align()) | MALLOCX_ARENA(ind.to_usize().unwrap()),
            )
        };
        assert_ne!(addr, ptr::null_mut());
        unsafe {
            *(addr as *mut char) = 'c';
            assert_eq!(*(addr as *mut char), 'c');
            debug!("addr : {}, char : {}", addr as usize, *(addr as *mut char));
        }
        thread.join().unwrap();
    }
}
