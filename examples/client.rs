//! This demo shows how to establish a connection between server and client
//! and the usage of rdma `read`, `write` and `send&recv` APIs.
//!
//! You can try this example by running:
//!
//!     cargo run --example server
//!
//! And then start client in another terminal by running:
//!
//!     cargo run --example client

use async_rdma::{LocalMrWriteAccess, Rdma};
use std::{alloc::Layout, time::Duration};
use tracing::debug;

async fn send_lmr_to_server(rdma: &Rdma) {
    let mut lmr = rdma.alloc_local_mr(Layout::new::<char>()).unwrap();
    unsafe { *(lmr.as_mut_ptr() as *mut char) = 'h' };
    rdma.send_local_mr(lmr).await.unwrap();
}

async fn request_then_write(rdma: &Rdma) {
    let mut rmr = rdma.request_remote_mr(Layout::new::<char>()).await.unwrap();
    let mut lmr = rdma.alloc_local_mr(Layout::new::<char>()).unwrap();
    unsafe { *(lmr.as_mut_ptr() as *mut char) = 'e' };
    rdma.write(&lmr, &mut rmr).await.unwrap();
    rdma.send_remote_mr(rmr).await.unwrap();
}

async fn send_data_to_server(rdma: &Rdma) {
    let mut lmr = rdma.alloc_local_mr(Layout::new::<char>()).unwrap();
    unsafe { *(lmr.as_mut_ptr() as *mut char) = 'y' };
    rdma.send(&lmr).await.unwrap();
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    debug!("client start");
    let rdma = Rdma::connect("127.0.0.1:5555", 1, 1, 64).await.unwrap();
    send_lmr_to_server(&rdma).await;
    // `send` is faster than `receive` because the workflow of `receive` operation is
    // more complex. If we run server and client in the same machine like this test and
    // `send` without `sleep`, the receiver will be too busy to response sender.
    // So the sender's RDMA netdev will retry again and again which make the situation worse.
    // You can skip this `sleep` if your receiver's machine is fast enough.
    tokio::time::sleep(Duration::from_millis(100)).await;
    request_then_write(&rdma).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    send_data_to_server(&rdma).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    debug!("client done");
}
