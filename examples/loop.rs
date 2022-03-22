use async_rdma::LocalMrReadAccess;
use async_rdma::MrAccess;
use async_rdma::Rdma;
use async_rdma::RdmaListener;
use portpicker::pick_unused_port;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use std::{
    alloc::Layout,
    io,
    net::{Ipv4Addr, SocketAddrV4},
};

async fn client(addr: SocketAddrV4) -> io::Result<()> {
    let rdma = Arc::new(Rdma::connect(addr, 1, 1, 64).await?);
    // send the content of lmr to server
    let mut handles = vec![];
    let mut thread_handles = vec![];
    let out = String::from_utf8(
        Command::new("sh")
            .arg("-c")
            .arg("prlimit")
            .output()
            .expect("failed to execute process")
            .stdout,
    )
    .unwrap();
    println!("ulimit info {}", out);
    for i in 0..1 {
        let rdma_clone = rdma.clone();
        let h =
            tokio::spawn(async move { rdma_clone.alloc_local_mr(Layout::new::<i32>()).unwrap() });
        thread_handles.push(h);
        println!("spawn to alloc  {}", i);
    }
    for _ in 0..1 {
        let rdma_clone = rdma.clone();
        tokio::time::sleep(Duration::from_millis(1)).await;
        handles.push(tokio::spawn(async move {
            let lm = rdma_clone.alloc_local_mr(Layout::new::<i32>()).unwrap();
            println!("alloc mr");
            unsafe { *(lm.as_ptr() as *mut i32) = 5 };
            rdma_clone
                .send(&lm)
                .await
                .map_err(|e| println!("{}", e))
                .unwrap();
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }
    for handle in thread_handles {
        handle.await.unwrap();
    }
    Ok(())
}

#[tokio::main]
async fn server(addr: SocketAddrV4) -> io::Result<()> {
    let rdma_listener = RdmaListener::bind(addr).await?;
    let rdma = Arc::new(rdma_listener.accept(1, 1, 64).await?);
    // receive the data sent by client and put it into an mr
    let mut handles = vec![];
    for _ in 0..1 {
        let rdma_clone = rdma.clone();
        handles.push(tokio::spawn(async move {
            let lm = rdma_clone.receive().await.unwrap();
            assert_eq!(unsafe { *(lm.as_ptr() as *mut i32) }, 5);
            assert_eq!(lm.length(), 4);
        }));
    }
    // wait for the agent thread to send all reponses to the remote.
    tokio::time::sleep(Duration::from_secs(5)).await;
    for (i, handle) in handles.into_iter().enumerate() {
        handle.await?;
        println!("done {}", i);
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let addr = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), pick_unused_port().unwrap());
    std::thread::spawn(move || server(addr));
    tokio::time::sleep(Duration::from_secs(2)).await;
    client(addr)
        .await
        .map_err(|err| println!("{}", err))
        .unwrap();
}
