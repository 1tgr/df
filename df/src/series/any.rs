use crate::{Series, VectorAny};
use crate::simd::Select;
use crate::storage::SimdStorage;

impl VectorAny for &Series<bool> {
    fn any(self) -> bool {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        self.data
            .try_fold_packed(index_exists, (), |(), item, mask| {
                if mask.select(item, 0).any() {
                    Err(true)
                } else {
                    Ok(())
                }
            })
            .err()
            .unwrap_or(false)
    }

    fn all(self) -> bool {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        self.data
            .try_fold_packed(index_exists, (), |(), item, mask| {
                if mask.select(item, !0).all() {
                    Ok(())
                } else {
                    Err(false)
                }
            })
            .err()
            .unwrap_or(true)
    }

    fn none(self) -> bool {
        !self.any()
    }
}
