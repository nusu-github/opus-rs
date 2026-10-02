/// Computes the integer logarithm base 2 of a value
/// Returns floor(log2(x)) + 1 for positive values, and zero otherwise.
pub(crate) fn ilog(x: isize) -> isize {
    if x <= 0 {
        return 0;
    }
    isize::BITS as isize - x.leading_zeros() as isize
}

pub(crate) fn sign(value: i32) -> i32 {
    match value.cmp(&0) {
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Greater => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::ilog;

    #[test]
    fn integer_log_uses_the_target_pointer_width() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(-1), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(255), 8);
        assert_eq!(ilog(isize::MAX), isize::BITS as isize - 1);
    }
}
