//! SigmaOS component binary: ml-dl

use sigmaos_ml-dl::Component;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Component::new().operate()?;
    println!("sigmaos-ml-dl: ok");
    Ok(())
}
