use crate::{Element, Series, Storage, VectorWhere, VectorWhereOr};

impl<T> VectorWhere<Series<bool>> for Series<T>
where
    T: Element + ?Sized,
{
    type Output = Self;

    fn where_(self, condition: Series<bool>) -> Self {
        let (index, data, condition_data) = self.into_aligned(condition);
        let condition_data = condition_data.as_vec(false);
        let (_index_data, index_exists) = index.as_vec_mask();
        let data = data.where_scalar(index_exists, condition_data.as_ref(), None);
        Series::new(index, data)
    }

    fn mask(self, condition: Series<bool>) -> Self {
        let (index, data, condition_data) = self.into_aligned(condition);
        let condition_data = condition_data.as_vec(false);
        let (_index_data, index_exists) = index.as_vec_mask();
        let data = data.mask_scalar(index_exists, condition_data.as_ref(), None);
        Series::new(index, data)
    }
}

impl<T> VectorWhereOr<Series<bool>, T> for Series<T>
where
    T: Element,
{
    type Output = Self;

    fn where_or(self, condition: Series<bool>, other: T) -> Self {
        self.where_or(condition, &other)
    }

    fn mask_or(self, condition: Series<bool>, other: T) -> Self {
        self.mask_or(condition, &other)
    }
}

impl<T> VectorWhereOr<Series<bool>, &T> for Series<T>
where
    T: Element + ?Sized,
{
    type Output = Self;

    fn where_or(self, condition: Series<bool>, other: &T) -> Self {
        let (index, data, condition_data) = self.into_aligned(condition);
        let condition_data = condition_data.as_vec(false);
        let (_index_data, index_exists) = index.as_vec_mask();
        let data = data.where_scalar(index_exists, condition_data.as_ref(), Some(other));
        Series::new(index, data)
    }

    fn mask_or(self, condition: Series<bool>, other: &T) -> Self {
        let (index, data, condition_data) = self.into_aligned(condition);
        let condition_data = condition_data.as_vec(false);
        let (_index_data, index_exists) = index.as_vec_mask();
        let data = data.mask_scalar(index_exists, condition_data.as_ref(), Some(other));
        Series::new(index, data)
    }
}

impl<T> VectorWhereOr<Series<bool>, Series<T>> for Series<T>
where
    T: Element + ?Sized,
{
    type Output = Self;

    fn where_or(self, condition: Series<bool>, other: Self) -> Self {
        let (index, data, condition_data, other_data) = self.into_aligned3(condition, other);
        let condition_data = condition_data.as_vec(false);
        let (_index_data, index_exists) = index.as_vec_mask();
        let data = data.where_(index_exists, condition_data.as_ref(), other_data);
        Series::new(index, data)
    }

    fn mask_or(self, condition: Series<bool>, other: Self) -> Self {
        other.where_or(condition, self)
    }
}
