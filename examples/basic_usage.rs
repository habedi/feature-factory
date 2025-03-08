// Run `make testdata` before running this example to download the data
// Run `cargo run --example basic_usage` to execute this example

use std::error::Error;
mod shared;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Construct the full path to the dataset
    let path = format!("{}/{}", shared::DATA_DIR, "yellow_tripdata_2019-01.parquet");

    // Load the dataset
    let input_df = shared::load_data(&path).await?;

    // Show the first 5 rows of the DataFrame
    input_df.limit(0, Some(5))?.show().await?;

    Ok(())
}
