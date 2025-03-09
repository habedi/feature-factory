//! ## Logging Configuration
//!
//! This module sets up logging automatically at program startup using the `ctor` crate.
//! Logging behavior is controlled by the `DEBUG_FEATURE_FACTORY` environment variable:
//!
//! - **Disabled** (default): If the variable is unset, empty, or explicitly set to `"0"` or `"false"`,
//!   no logging will be initialized.
//! - **Enabled**: Any other value enables logging with a maximum log level of `DEBUG`.
//!
//! ### Usage Example
//!
//! To enable debug-level logging, set the environment variable before running your application:
//!
//! ```sh
//! export DEBUG_FEATURE_FACTORY=true
//! ```

use ctor::ctor;
use tracing::Level;

#[ctor]
fn set_debug_level() {
    let logging_disabled = std::env::var("DEBUG_FEATURE_FACTORY")
        .map_or(true, |v| v == "0" || v == "false" || v.is_empty());

    if !logging_disabled {
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .init();
    }
}
