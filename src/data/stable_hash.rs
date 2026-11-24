//! A hasher whose output is the same in every process and on every platform.
//!
//! [`Subject::hash`](crate::Subject::hash) seeds the SDE random streams, so it
//! must not depend on a per-process key (as `ahash`'s default does), on
//! endianness or on pointer width.

use std::hash::Hasher;

/// FNV-1a over little-endian bytes, finished with the SplitMix64 finalizer.
pub(crate) struct StableHasher(u64);

impl Default for StableHasher {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for StableHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u16(&mut self, i: u16) {
        self.write(&i.to_le_bytes());
    }

    fn write_u32(&mut self, i: u32) {
        self.write(&i.to_le_bytes());
    }

    fn write_u64(&mut self, i: u64) {
        self.write(&i.to_le_bytes());
    }

    fn write_u128(&mut self, i: u128) {
        self.write(&i.to_le_bytes());
    }

    // Widen so 32- and 64-bit targets agree.
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }

    fn write_isize(&mut self, i: isize) {
        self.write_u64(i as i64 as u64);
    }

    fn finish(&self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    #[test]
    fn integer_widths_hash_as_little_endian_u64() {
        let mut a = StableHasher::default();
        7usize.hash(&mut a);
        let mut b = StableHasher::default();
        7u64.hash(&mut b);
        assert_eq!(a.finish(), b.finish());
    }

    /// A change here changes every seeded SDE result.
    #[test]
    fn output_is_fixed() {
        let mut hasher = StableHasher::default();
        "pharmsol".hash(&mut hasher);
        42u64.hash(&mut hasher);
        assert_eq!(hasher.finish(), 13191298833039684293);
    }
}
