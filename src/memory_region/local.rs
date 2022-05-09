use super::{raw::RawMemoryRegion, MrAccess, MrToken};
use std::{
    fmt::Debug,
    io,
    ops::Range,
    slice,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tracing::debug;

use parking_lot::{lock_api::RawRwLock as RawRwLockTrait, RawRwLock};
/// Local memory region trait
pub trait LocalMrReadAccess: MrAccess {
    /// Get the start pointer
    ///
    /// Return None if the mr is being used by RDMA ops
    #[inline]
    #[allow(clippy::as_conversions)]
    fn try_as_ptr(&self) -> Option<*const u8> {
        if self.is_readable() {
            return Some(self.addr() as _);
        }
        None
    }

    /// Get the start pointer until it is readable
    ///
    /// If this mr is being used in RDMA ops, the thread may be blocked
    #[inline]
    #[allow(clippy::as_conversions)]
    fn as_ptr(&self) -> *const u8 {
        self.get_inner().lock.0.lock_shared();
        // Safety: locked before
        unsafe {
            self.get_inner().lock.0.unlock_shared();
        }
        self.addr() as _
    }

    /// Get the start pointer
    ///
    /// TODO: move unchecked methords to unsafe trait
    #[inline]
    #[allow(clippy::as_conversions)]
    fn as_ptr_unchecked(&self) -> *const u8 {
        self.addr() as _
    }

    /// Get the memory region as slice
    ///
    /// Return None if the mr is being used by RDMA ops
    #[inline]
    fn try_as_slice(&self) -> Option<&[u8]> {
        self.try_as_ptr()
            .map(|ptr| unsafe { slice::from_raw_parts(ptr, self.length()) })
    }

    /// Get the memory region as slice until it is readable
    ///
    /// If this mr is being used in RDMA ops, the thread may be blocked
    #[inline]
    #[allow(clippy::as_conversions)]
    fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.length()) }
    }

    /// Get the local key
    fn lkey(&self) -> u32;

    /// New a token with specified timeout
    #[inline]
    fn token_with_timeout(&self, timeout: Duration) -> Option<MrToken> {
        SystemTime::now().checked_add(timeout).map_or_else(
            || None,
            |ddl| {
                Some(MrToken {
                    addr: self.addr(),
                    len: self.length(),
                    rkey: self.rkey(),
                    ddl,
                })
            },
        )
    }
    /// Get the corresponding `LocalMrInner`
    fn get_inner(&self) -> &Arc<LocalMrInner>;

    /// Get the lock of the corresponding `LocalMrInner`
    #[inline]
    fn get_lock(&self) -> &MrRwLock {
        &self.get_inner().lock
    }

    /// Is the corresponding `LocalMrInner` readable?
    #[inline]
    fn is_readable(&self) -> bool {
        !self.get_inner().lock.0.is_locked_exclusive()
    }
    /// Is the corresponding `LocalMrInner` writeable?
    #[inline]
    fn is_writeable(&self) -> bool {
        !self.get_inner().lock.0.is_locked()
    }

    // TODO: we can impl new APIs that block current thread and wait until the mr
    // is idel by using `lock_exclusive/shared` without `try`.

    /// Try to lock the corresponding `LocalMrInner` to write
    #[inline]
    fn try_lock_exclusive(&self) -> bool {
        self.get_inner().lock.0.try_lock_exclusive()
    }

    /// Try to lock the corresponding `LocalMrInner` to read
    #[inline]
    fn try_lock_shared(&self) -> bool {
        self.get_inner().lock.0.try_lock_shared()
    }
}

/// Writable local mr trait
pub trait LocalMrWriteAccess: MrAccess + LocalMrReadAccess {
    /// Get the memory region start mut addr
    ///
    /// Return None if the mr is being used by RDMA ops
    #[inline]
    #[allow(clippy::as_conversions)]
    fn try_as_mut_ptr(&mut self) -> Option<*mut u8> {
        // const pointer to mut pointer is safe
        if self.is_writeable() {
            return Some(self.as_ptr_unchecked() as _);
        }
        None
    }

    /// Get the mutable start pointer until it is writeable
    ///
    /// If this mr is being used in RDMA ops, the thread may be blocked
    #[inline]
    #[allow(clippy::as_conversions)]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.get_inner().lock.0.lock_exclusive();
        // Safety: locked before
        unsafe {
            self.get_inner().lock.0.unlock_exclusive();
        }
        self.addr() as _
    }

    /// Get the memory region as mut slice
    ///
    /// Return None if the mr is being used by RDMA ops
    #[inline]
    fn try_as_mut_slice(&mut self) -> Option<&mut [u8]> {
        self.try_as_mut_ptr()
            .map(|ptr| return unsafe { slice::from_raw_parts_mut(ptr, self.length()) })
    }

    /// Get the memory region as mutable slice until it is writeable
    ///
    /// If this mr is being used in RDMA ops, the thread may be blocked
    #[inline]
    #[allow(clippy::as_conversions)]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.length()) }
    }
}

/// Local Memory Region
#[derive(Debug)]
pub struct LocalMr {
    /// Local Memory Region Inner
    inner: Arc<LocalMrInner>,
    /// The start address of this mr
    addr: usize,
    /// the length of this mr
    len: usize,
}

impl MrAccess for LocalMr {
    #[inline]
    fn addr(&self) -> usize {
        self.addr
    }

    #[inline]
    fn length(&self) -> usize {
        self.len
    }

    #[inline]
    fn rkey(&self) -> u32 {
        self.inner.rkey()
    }
}

impl LocalMrReadAccess for LocalMr {
    #[inline]
    fn lkey(&self) -> u32 {
        self.inner.lkey()
    }

    #[inline]
    fn get_inner(&self) -> &Arc<LocalMrInner> {
        &self.inner
    }
}

impl LocalMrWriteAccess for LocalMr {}

impl LocalMr {
    /// New Local Mr
    pub(crate) fn new(inner: Arc<LocalMrInner>) -> Self {
        Self {
            addr: inner.addr,
            len: inner.len,
            inner,
        }
    }

    /// Get a local mr slice
    #[inline]
    pub fn get(&self, i: Range<usize>) -> io::Result<LocalMrSlice> {
        // SAFETY: `self` is checked to be valid and in bounds above.
        if i.start >= i.end || i.end > self.len {
            Err(io::Error::new(io::ErrorKind::Other, "wrong range of lmr"))
        } else {
            Ok(LocalMrSlice::new(
                self,
                Arc::<LocalMrInner>::clone(&self.inner),
                self.addr().wrapping_add(i.start),
                i.len(),
            ))
        }
    }

    /// Get a mutable local mr slice
    #[inline]
    pub fn get_mut(&mut self, i: Range<usize>) -> io::Result<LocalMrSliceMut> {
        // SAFETY: `self` is checked to be valid and in bounds above.
        if i.start >= i.end || i.end > self.length() {
            Err(io::Error::new(io::ErrorKind::Other, "wrong range of lmr"))
        } else {
            Ok(LocalMrSliceMut::new(
                self,
                Arc::<LocalMrInner>::clone(&self.inner),
                self.addr().wrapping_add(i.start),
                i.len(),
            ))
        }
    }

    /// take the ownership and return a sub local mr from self.
    #[inline]
    pub(crate) fn take(mut self, i: Range<usize>) -> io::Result<Self> {
        // SAFETY: `self` is checked to be valid and in bounds above.
        if i.start >= i.end || i.end > self.length() {
            Err(io::Error::new(io::ErrorKind::Other, "wrong range of lmr"))
        } else {
            self.addr = self.addr.wrapping_add(i.start);
            self.len = i.end.wrapping_sub(i.start);
            Ok(self)
        }
    }
}

/// Read-Write-Lock of `LocalMrInner`
pub struct MrRwLock(RawRwLock);

impl Debug for MrRwLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("MrState")
            .field(&self.0.is_locked())
            .field(&self.0.is_locked_exclusive())
            .finish()
    }
}

impl MrRwLock {
    /// Create a new `MrRwLock`
    fn new() -> Self {
        MrRwLock(RawRwLock::INIT)
    }
}

/// Local Memory Region inner
#[derive(Debug)]
pub struct LocalMrInner {
    /// The start address of this mr
    addr: usize,
    /// The length of this mr
    len: usize,
    /// The raw mr where this local mr comes from.
    raw: Arc<RawMemoryRegion>,
    /// `RwLock` of this lcoal memory region
    lock: MrRwLock,
}

impl Drop for LocalMrInner {
    #[inline]
    #[allow(clippy::as_conversions)]
    fn drop(&mut self) {
        debug!("drop LocalMr {:?}", self);
        unsafe { tikv_jemalloc_sys::free(self.addr as _) }
    }
}

impl MrAccess for LocalMrInner {
    #[inline]
    fn addr(&self) -> usize {
        self.addr
    }

    #[inline]
    fn length(&self) -> usize {
        self.len
    }

    #[inline]
    fn rkey(&self) -> u32 {
        self.raw.lkey()
    }
}

impl LocalMrInner {
    /// New Local Mr
    pub(crate) fn new(addr: usize, len: usize, raw: Arc<RawMemoryRegion>) -> Self {
        Self {
            addr,
            len,
            raw,
            lock: MrRwLock::new(),
        }
    }

    /// Get local key of memory region
    fn lkey(&self) -> u32 {
        self.raw.lkey()
    }

    /// Unlock after the RDMA ops done
    pub(crate) fn unlock(&self) {
        // SAFETY: the lock is locked before RDMA ops
        if self.lock.0.is_locked_exclusive() {
            unsafe { self.lock.0.unlock_exclusive() }
        } else {
            unsafe { self.lock.0.unlock_shared() }
        }
    }
}

impl MrAccess for &LocalMr {
    #[inline]
    fn addr(&self) -> usize {
        self.addr
    }

    #[inline]
    fn length(&self) -> usize {
        self.len
    }

    #[inline]
    fn rkey(&self) -> u32 {
        self.inner.rkey()
    }
}

impl LocalMrReadAccess for &LocalMr {
    #[inline]
    fn lkey(&self) -> u32 {
        self.inner.lkey()
    }

    #[inline]
    fn get_inner(&self) -> &Arc<LocalMrInner> {
        &self.inner
    }
}

/// A slice of `LocalMr`
#[derive(Debug)]
pub struct LocalMrSlice<'a> {
    /// The local mr where this local mr slice comes from.
    lmr: &'a LocalMr,
    /// The local mr where this local mr slice comes from.
    inner: Arc<LocalMrInner>,
    /// The start address of this mr
    addr: usize,
    /// the length of this mr
    len: usize,
}

impl MrAccess for LocalMrSlice<'_> {
    #[inline]
    fn addr(&self) -> usize {
        self.addr
    }

    #[inline]
    fn length(&self) -> usize {
        self.len
    }

    #[inline]
    fn rkey(&self) -> u32 {
        self.lmr.rkey()
    }
}

impl LocalMrReadAccess for LocalMrSlice<'_> {
    fn lkey(&self) -> u32 {
        self.lmr.lkey()
    }

    #[inline]
    fn get_inner(&self) -> &Arc<LocalMrInner> {
        &self.inner
    }
}

impl<'a> LocalMrSlice<'a> {
    /// New a local mr slice.
    pub(crate) fn new(lmr: &'a LocalMr, inner: Arc<LocalMrInner>, addr: usize, len: usize) -> Self {
        Self {
            lmr,
            inner,
            addr,
            len,
        }
    }
}

/// Mutable local mr slice
#[derive(Debug)]
pub struct LocalMrSliceMut<'a> {
    /// The local mr where this local mr slice comes from.
    lmr: &'a mut LocalMr,
    /// The local mr where this local mr slice comes from.
    inner: Arc<LocalMrInner>,
    /// The start address of this mr
    addr: usize,
    /// the length of this mr
    len: usize,
}

impl<'a> LocalMrSliceMut<'a> {
    /// New a mutable local mr slice.
    pub(crate) fn new(
        lmr: &'a mut LocalMr,
        inner: Arc<LocalMrInner>,
        addr: usize,
        len: usize,
    ) -> Self {
        Self {
            lmr,
            inner,
            addr,
            len,
        }
    }
}

impl MrAccess for LocalMrSliceMut<'_> {
    #[inline]
    fn addr(&self) -> usize {
        self.addr
    }

    #[inline]
    fn length(&self) -> usize {
        self.len
    }

    #[inline]
    fn rkey(&self) -> u32 {
        self.lmr.rkey()
    }
}

impl LocalMrReadAccess for LocalMrSliceMut<'_> {
    fn lkey(&self) -> u32 {
        self.lmr.lkey()
    }

    #[inline]
    fn get_inner(&self) -> &Arc<LocalMrInner> {
        &self.inner
    }
}

impl LocalMrWriteAccess for LocalMrSliceMut<'_> {}
