


pub use colored::*;
#[macro_export]
macro_rules! info{
    ($($arg:tt)*) => ({
        eprintln!("{} {}","[Info]".bright_green().bold(),std::format_args!($($arg)*));
    })
}

#[macro_export]
macro_rules! error{
    ($($arg:tt)*) => ({
        eprintln!("{} {}","[Error]".bright_red().bold(),std::format_args!($($arg)*));
    })
}

#[macro_export]
macro_rules! panic_error{
    ($($arg:tt)*) => ({
        panic!("{} {}","[Critical Error]".bright_red().bold(),std::format_args!($($arg)*));
    })
}