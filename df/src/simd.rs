use std::marker::PhantomData;
use std::slice;

use bit_vec::BitBlock;
use packed_simd::*;

pub trait AsFlat<A: SimdArray> {
    fn as_flat(&self) -> &[A::T];
}

impl<A: SimdArray> AsFlat<A> for [Simd<A>] {
    fn as_flat(&self) -> &[A::T] {
        let ptr = self.as_ptr() as *const A::T;
        let len = self.len() * A::N;
        unsafe { slice::from_raw_parts(ptr, len) }
    }
}

pub trait AsFlatMut<A: SimdArray> {
    fn as_flat_mut(&mut self) -> &mut [A::T];
}

impl<A: SimdArray> AsFlatMut<A> for [Simd<A>] {
    fn as_flat_mut(&mut self) -> &mut [A::T] {
        let ptr = self.as_mut_ptr() as *mut A::T;
        let len = self.len() * A::N;
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }
}

pub trait ToPacked<A: SimdArray> {
    fn to_packed(&self) -> Vec<Simd<A>>;
}

impl<A> ToPacked<A> for [A::T]
where
    A: AsMut<[<A as SimdArray>::T]> + Clone + Default + SimdArray + Into<Simd<A>>,
    A::T: Copy,
{
    fn to_packed(&self) -> Vec<Simd<A>> {
        let mut data: Vec<Simd<A>> = vec![A::default().into(); self.len() / A::N];
        let (flat_head, flat_tail) = self.split_at(data.len() * A::N);
        data.as_flat_mut().copy_from_slice(flat_head);

        let mut tail = A::default();
        tail.as_mut()[0..flat_tail.len()].copy_from_slice(flat_tail);
        data.push(tail.into());
        data
    }
}

pub trait FromBitBlock<T> {
    fn from_bit_block(block: T) -> Self;
}

impl FromBitBlock<u32> for u32 {
    fn from_bit_block(block: u32) -> u32 {
        block
    }
}

impl<T> FromBitBlock<T> for m8x32
where
    T: BitBlock + Into<u32>,
{
    fn from_bit_block(_block: T) -> Self {
        unimplemented!()
    }
}

impl<T> FromBitBlock<T> for m16x16
where
    T: BitBlock + Into<u32>,
{
    fn from_bit_block(_block: T) -> Self {
        unimplemented!()
    }
}

impl<T> FromBitBlock<T> for m32x8
where
    T: BitBlock + Into<u32>,
{
    fn from_bit_block(_block: T) -> Self {
        unimplemented!()
    }
}

const BITS_TO_M64X4: [m64x4; 16] = [
    m64x4::new(false, false, false, false),
    m64x4::new(true, false, false, false),
    m64x4::new(false, true, false, false),
    m64x4::new(true, true, false, false),
    m64x4::new(false, false, true, false),
    m64x4::new(true, false, true, false),
    m64x4::new(false, true, true, false),
    m64x4::new(true, true, true, false),
    m64x4::new(false, false, false, true),
    m64x4::new(true, false, false, true),
    m64x4::new(false, true, false, true),
    m64x4::new(true, true, false, true),
    m64x4::new(false, false, true, true),
    m64x4::new(true, false, true, true),
    m64x4::new(false, true, true, true),
    m64x4::new(true, true, true, true),
];

impl<T> FromBitBlock<T> for m64x4
where
    T: BitBlock + Into<u32>,
{
    fn from_bit_block(block: T) -> Self {
        assert!(T::bits() >= 4);
        BITS_TO_M64X4[(block % (T::one() << 4)).into() as usize]
    }
}

impl<T> FromBitBlock<T> for m32x4
where
    T: BitBlock + Into<u32>,
{
    fn from_bit_block(block: T) -> Self {
        m64x4::from_bit_block(block).into()
    }
}

impl<T> FromBitBlock<T> for m16x4
where
    T: BitBlock + Into<u32>,
{
    fn from_bit_block(block: T) -> Self {
        m64x4::from_bit_block(block).into()
    }
}

impl<T> FromBitBlock<T> for m8x4
where
    T: BitBlock + Into<u32>,
{
    fn from_bit_block(block: T) -> Self {
        m64x4::from_bit_block(block).into()
    }
}

pub trait Select<T> {
    fn select(self, a: T, b: T) -> T;
}

impl Select<u32> for u32 {
    fn select(self, a: u32, b: u32) -> u32 {
        (self & a) | (!self & b)
    }
}

impl<A: SimdArray> Select<Simd<A>> for m8x32
where
    A: SimdArray<NT = [u32; 32]>,
{
    fn select(self, a: Simd<A>, b: Simd<A>) -> Simd<A> {
        Self::select(self, a, b)
    }
}

impl<A: SimdArray> Select<Simd<A>> for m16x16
where
    A: SimdArray<NT = [u32; 16]>,
{
    fn select(self, a: Simd<A>, b: Simd<A>) -> Simd<A> {
        Self::select(self, a, b)
    }
}

impl<A: SimdArray> Select<Simd<A>> for m32x8
where
    A: SimdArray<NT = [u32; 8]>,
{
    fn select(self, a: Simd<A>, b: Simd<A>) -> Simd<A> {
        Self::select(self, a, b)
    }
}

impl<A: SimdArray> Select<Simd<A>> for m8x4
where
    A: SimdArray<NT = [u32; 4]>,
{
    fn select(self, a: Simd<A>, b: Simd<A>) -> Simd<A> {
        Self::select(self, a, b)
    }
}

impl<A: SimdArray> Select<Simd<A>> for m16x4
where
    A: SimdArray<NT = [u32; 4]>,
{
    fn select(self, a: Simd<A>, b: Simd<A>) -> Simd<A> {
        Self::select(self, a, b)
    }
}

impl<A: SimdArray> Select<Simd<A>> for m32x4
where
    A: SimdArray<NT = [u32; 4]>,
{
    fn select(self, a: Simd<A>, b: Simd<A>) -> Simd<A> {
        Self::select(self, a, b)
    }
}

impl<A: SimdArray> Select<Simd<A>> for m64x4
where
    A: SimdArray<NT = [u32; 4]>,
{
    fn select(self, a: Simd<A>, b: Simd<A>) -> Simd<A> {
        Self::select(self, a, b)
    }
}

pub trait Splat<T> {
    fn splat(a: T) -> Self;
}

pub trait VectorEq<Rhs, Output> {
    fn eq(self, rhs: Rhs) -> Output;
    fn ne(self, rhs: Rhs) -> Output;
}

impl<T> VectorEq<T, bool> for T
where
    T: PartialEq<T>,
{
    fn eq(self, rhs: T) -> bool {
        self == rhs
    }

    fn ne(self, rhs: T) -> bool {
        self != rhs
    }
}

impl VectorEq<u32, u32> for u32 {
    fn eq(self, _rhs: u32) -> u32 {
        unimplemented!()
    }

    fn ne(self, _rhs: u32) -> u32 {
        unimplemented!()
    }
}

pub trait VectorCmp<Rhs, Output> {
    fn lt(self, rhs: Rhs) -> Output;
    fn lte(self, rhs: Rhs) -> Output;
    fn gt(self, rhs: Rhs) -> Output;
    fn gte(self, rhs: Rhs) -> Output;
}

impl<T> VectorCmp<T, bool> for T
where
    T: PartialOrd<T>,
{
    fn lt(self, rhs: T) -> bool {
        self < rhs
    }

    fn lte(self, rhs: T) -> bool {
        self <= rhs
    }

    fn gt(self, rhs: T) -> bool {
        self > rhs
    }

    fn gte(self, rhs: T) -> bool {
        self >= rhs
    }
}

impl VectorCmp<u32, u32> for u32 {
    fn lt(self, _rhs: u32) -> u32 {
        unimplemented!()
    }

    fn lte(self, _rhs: u32) -> u32 {
        unimplemented!()
    }

    fn gt(self, _rhs: u32) -> u32 {
        unimplemented!()
    }

    fn gte(self, _rhs: u32) -> u32 {
        unimplemented!()
    }
}

pub trait VectorSum<T> {
    fn sum(self) -> T;
}

impl VectorSum<f64> for f64x4 {
    fn sum(self) -> f64 {
        f64x4::sum(self)
    }
}

impl VectorSum<i64> for i64x4 {
    fn sum(self) -> i64 {
        i64x4::wrapping_sum(self)
    }
}

pub trait VectorAny {
    fn any(self) -> bool;
    fn all(self) -> bool;
    fn none(self) -> bool;
}

impl VectorAny for u32 {
    fn any(self) -> bool {
        self != 0
    }

    fn all(self) -> bool {
        self == !0
    }

    fn none(self) -> bool {
        self == 0
    }
}

pub trait VectorWhere<B> {
    type Output;

    fn where_(self, condition: B) -> Self::Output;
    fn mask(self, condition: B) -> Self::Output;
}

pub trait VectorWhereOr<B, T> {
    type Output;

    fn where_or(self, condition: B, other: T) -> Self::Output;
    fn mask_or(self, condition: B, other: T) -> Self::Output;
}

pub struct Masks<I: Iterator, M> {
    bit_blocks: I,
    block: I::Item,
    bits: usize,
    _pd: PhantomData<M>,
}

impl<I, MA> Iterator for Masks<I, Simd<MA>>
where
    I: Iterator,
    I::Item: BitBlock,
    MA: SimdArray,
    Simd<MA>: FromBitBlock<I::Item>,
{
    type Item = Simd<MA>;

    fn next(&mut self) -> Option<Self::Item> {
        let (block, bits) = match self.bits {
            0 => (self.bit_blocks.next()?, I::Item::bits()),
            bits => (self.block, bits),
        };

        self.bits = bits - MA::N;
        self.block = block >> MA::N;
        Some(Simd::from_bit_block(block))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.bit_blocks.size_hint();
        let lo = (lo * I::Item::bits()) / MA::N;
        let hi = hi.map(|hi| (hi * I::Item::bits()) / MA::N);
        (lo, hi)
    }
}

pub trait IterMasks<M>
where
    Self: Iterator + Sized,
{
    fn masks(self) -> Masks<Self, M>;
}

impl<T, MA> IterMasks<Simd<MA>> for T
where
    Self: Iterator + Sized,
    Self::Item: BitBlock,
    MA: SimdArray,
    Simd<MA>: FromBitBlock<Self::Item>,
{
    fn masks(self) -> Masks<Self, Simd<MA>> {
        assert!(Self::Item::bits() >= MA::N);
        assert_eq!(Self::Item::bits() % MA::N, 0);

        Masks {
            bit_blocks: self,
            block: Self::Item::zero(),
            bits: 0,
            _pd: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bit_vec::BitVec;

    #[test]
    fn test_masks_u8() {
        let v = vec![false, true, true, false, true].into_iter().collect::<BitVec<u8>>();
        let blocks = v.blocks();
        assert_eq!(1, blocks.len());

        let masks = blocks.masks().collect::<Vec<m64x4>>();
        let expected = vec![
            m64x4::new(false, true, true, false),
            m64x4::new(true, false, false, false),
        ];
        assert_eq!(expected, masks);
    }

    #[test]
    fn test_masks_u32() {
        let v = vec![false, true, true, false, true].into_iter().collect::<BitVec>();
        let blocks = v.blocks();
        assert_eq!(1, blocks.len());

        let masks = blocks.masks().collect::<Vec<m64x4>>();
        let mut expected = vec![
            m64x4::new(false, true, true, false),
            m64x4::new(true, false, false, false),
        ];
        expected.resize(32 / 4, m64x4::new(false, false, false, false));
        assert_eq!(expected, masks);
    }
}
