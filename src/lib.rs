//   Copyright 2017 Torsten Weber
//
//   Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
//   http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
//   <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your option.
//   This file may not be copied, modified, or distributed except according
//   to those terms.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! A library for working with four-vectors (or Lorentz vectors).
//!
//! This library is mostly aimed at high energy physics applications,
//! but should be general enough for usage in special relativity.
//!
//! # Design
//!
//! The library is intended to be usable for both space-time vectors and
//! four-momenta.  Therefore this library includes functions and methods
//! with the common names for both usecases.
//!
//! The design of this library is strongly inspired by the class
//! [`TLorentzVector`] from the C++ particle physics data analysis
//! framework [ROOT]. However it differs from `TLorentzVector in several
//! key aspects which are listed in the [Comparison with TLorentzVector]
//! section.
//!
//! [`TLorentzVector`]: https://root.cern.ch/doc/master/classTLorentzVector.html
//! [ROOT]: https://root.cern.ch

#![deny(missing_docs)]

#[cfg(test)]
#[macro_use]
extern crate approx;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use std::ops::{Index, IndexMut};

/// The four components of a LorentzVector.
///
/// This enum is used to index a LorentzVector.
///
/// # Examples
///
/// ```rust
/// use lorentz_vector::{Component, LorentzVector};
///
/// let mut vector = LorentzVector::with_epxpypz(40., 30., 20., 10.);
/// assert_eq!(vector[Component::Zero], vector.e());
/// assert_eq!(vector[Component::One], vector.px());
/// assert_eq!(vector[Component::Two], vector.py());
/// assert_eq!(vector[Component::Three], vector.pz());
///
/// assert_eq!(vector.py(), 20.);
/// vector[Component::Two] -= 5.;
/// assert_eq!(vector.py(), 15.);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Component {
    /// Index of the time or energy component of a LorentzVector.
    Zero,
    /// Index of the x or px component of a LorentzVector.
    One,
    /// Index of the y or py component of a LorentzVector.
    Two,
    /// Index of the z or pz component of a LorentzVector.
    Three,
}

/// The LorentzVector type.
///
/// A `LorentzVector` can be used to describe a vector in space time
/// with coordinates `t`, `x`, `y`, `z` or a four-momentum with
/// coordinates `E`, `px`, `py` and `pz`.
///
/// # Creating LorentzVectors
///
/// There are three types of functions to create LorentzVectors:
///
/// * [`new`] and [`default`] that create a four-vector with all
/// components equal to zero.  * [`with_txyz`] and [`with_epxpypz`] to
/// create four-vectors with given time- and space-like components.  *
/// [`with_mpxpypz`] to create a four-momentum with given mass and
/// three-momentum.
///
/// [`new`]: struct.LorentzVector.html#method.new
/// [`default`]: https://doc.rust-lang.org/nightly/core/default/trait.Default.html#tymethod.default
/// [`with_txyz`]: struct.LorentzVector.html#method.with_txyz
/// [`with_epxpypz`]: struct.LorentzVector.html#method.with_epxpypz
/// [`with_mpxpypz`]: struct.LorentzVector.html#method.with_mpxpypz
///
/// # Accessing and modifying components of a LorentzVector
///
/// Components of a LorentzVector can be accessed either using getter
/// functions or by indexing the vector.  The components can be modified
/// using mutable getters or again by indexing the vector.
///
/// ## Getters
///
/// The getters [`t`], [`x`], [`y`], [`z`] and their aliases [`e`],
/// [`px`], [`py`], [`pz`] can be used to obtain the components of a
/// `LorentzVector`.
///
/// ```rust
/// use lorentz_vector::LorentzVector;
///
/// let vector = LorentzVector::with_txyz(4.0, -2.1, 9e-8, 12.03);
/// assert_eq!(vector.t(), 4.0);
/// assert_eq!(vector.px(), -2.1);
/// assert_eq!(vector.py(), 9e-8);
/// assert_eq!(vector.z(), 12.03);
/// ```
///
/// [`t`]: struct.LorentzVector.html#method.t
/// [`x`]: struct.LorentzVector.html#method.x
/// [`y`]: struct.LorentzVector.html#method.y
/// [`z`]: struct.LorentzVector.html#method.z
/// [`e`]: struct.LorentzVector.html#method.e
/// [`px`]: struct.LorentzVector.html#method.px
/// [`py`]: struct.LorentzVector.html#method.py
/// [`pz`]: struct.LorentzVector.html#method.pz
///
/// ## Mutable getters
///
/// The mutable getters [`t_mut`], [`x_mut`], [`y_mut`], [`z_mut`] and
/// their aliases [`e_mut`], [`px_mut`], [`py_mut`], [`pz_mut`] can be
/// used to modify the components of a `LorentzVector`.
///
/// ```rust
/// use lorentz_vector::LorentzVector;
///
///
/// let original = LorentzVector::with_txyz(4.0, -2.1, 9e-8, 12.03);
/// let mut repro = LorentzVector::new();
/// *repro.t_mut() = 4.0;
/// *repro.px_mut() -= 2.1;
/// *repro.py_mut() = 9e-8;
/// *repro.z_mut() += 12.03;
/// assert_eq!(original, repro);
/// ```
///
/// [`t_mut`]: struct.LorentzVector.html#method.t_mut
/// [`x_mut`]: struct.LorentzVector.html#method.x_mut
/// [`y_mut`]: struct.LorentzVector.html#method.y_mut
/// [`z_mut`]: struct.LorentzVector.html#method.z_mut
/// [`e_mut`]: struct.LorentzVector.html#method.e_mut
/// [`px_mut`]: struct.LorentzVector.html#method.px_mut
/// [`py_mut`]: struct.LorentzVector.html#method.py_mut
/// [`pz_mut`]: struct.LorentzVector.html#method.pz_mut
///
/// ## Indexing a LorentzVector
///
/// A `LorentzVector` can be indexed by a [`Component`]. As usual, the
/// time-like component has index `Zero`, while the `x`, `y` and `z`
/// components have indices `One`, `Two`, and `Three` respectively.
///
/// ```rust
/// use lorentz_vector::{Component, LorentzVector};
///
/// let mut vector = LorentzVector::with_epxpypz(40., 30., 20., 10.);
/// assert_eq!(vector[Component::Zero], vector.e());
/// assert_eq!(vector[Component::One], vector.px());
/// assert_eq!(vector[Component::Two], vector.py());
/// assert_eq!(vector[Component::Three], vector.pz());
///
/// assert_eq!(vector.py(), 20.);
/// vector[Component::Two] -= 5.;
/// assert_eq!(vector.py(), 15.);
/// ```
///
/// [`Component`]: enum.Component.html
///
///
/// # Arithmetic operations
///
/// `LorentzVector` supports addition and subtraction with other
/// `LorentzVectors` and multiplication and division with `f64`.
///
/// ```rust
/// use lorentz_vector::LorentzVector;
///
/// let a = LorentzVector::with_txyz(1., 2., 3., 4.);
/// let b = LorentzVector::with_txyz(-1., -2., -3., -4.);
///
/// assert_eq!(a+b, LorentzVector::new());
/// let div = a-b;
/// assert_eq!(div, LorentzVector::with_txyz(2., 4., 6., 8.));
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LorentzVector {
    e: f64,
    px: f64,
    py: f64,
    pz: f64,
}
impl LorentzVector {
    /// Creates a LorentzVector with all components set to zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lorentz_vector::LorentzVector;
    ///
    /// let zero_vector = LorentzVector::new();
    /// assert_eq!(zero_vector.t(), 0.);
    /// assert_eq!(zero_vector.x(), 0.);
    /// assert_eq!(zero_vector.y(), 0.);
    /// assert_eq!(zero_vector.z(), 0.);
    /// ```
    pub fn new() -> LorentzVector {
        LorentzVector { e: 0., px: 0., py: 0., pz: 0. }
    }

    /// Creates a LorentzVector with given `t`, `x`, `y` and `z`
    /// components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lorentz_vector::LorentzVector;
    ///
    /// let vector = LorentzVector::with_txyz(4.0, -2.1, 9e-8, 12.03);
    /// assert_eq!(vector.t(), 4.0);
    /// assert_eq!(vector.x(), -2.1);
    /// assert_eq!(vector.y(), 9e-8);
    /// assert_eq!(vector.z(), 12.03);
    /// ```
    pub fn with_txyz(t: f64, x: f64, y: f64, z: f64) -> LorentzVector {
        LorentzVector { e: t, px: x, py: y, pz: z }
    }

    /// Creates a LorentzVector with given `e`, `px`, `py` and `pz`
    /// components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lorentz_vector::LorentzVector;
    ///
    /// let vector = LorentzVector::with_epxpypz(80.2, -6.3e-2, 8.0e-1, 7.319e1);
    /// assert_eq!(vector.e(), 80.2);
    /// assert_eq!(vector.px(), -6.3e-2);
    /// assert_eq!(vector.py(), 8.0e-1);
    /// assert_eq!(vector.pz(), 7.319e1);
    /// ```
    pub fn with_epxpypz(e: f64, px: f64, py: f64, pz: f64) -> LorentzVector {
        LorentzVector { e, px, py, pz }
    }

    /// Creates a LorentzVector with given mass and three-momentum.
    ///
    /// # Panics
    ///
    /// If the mass `m` is negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[macro_use]
    /// extern crate approx;
    ///
    /// extern crate lorentz_vector;
    /// use lorentz_vector::LorentzVector;
    ///
    /// fn main() {
    ///     let top = LorentzVector::with_mpxpypz(173.2, 0., 103., 300.);
    ///     assert_relative_eq!(top.mass(), 173.2, max_relative=1e-10);
    /// }
    /// ```
    pub fn with_mpxpypz(m: f64, px: f64, py: f64, pz: f64) -> LorentzVector {
        if m < 0. {
            panic!("LorentzVector::with_mpxpypz: Negative mass ({})", m);
        }
        let e_sq = m.powi(2) + px.powi(2) + py.powi(2) + pz.powi(2);
        let e = e_sq.sqrt();
        LorentzVector { e, px, py, pz }
    }

    /// Returns the time component of the LorentzVector.
    pub fn t(&self) -> f64 {
        self.e
    }
    /// Returns the x component of the LorentzVector.
    pub fn x(&self) -> f64 {
        self.px
    }
    /// Returns the y component of the LorentzVector.
    pub fn y(&self) -> f64 {
        self.py
    }
    /// Returns the z component of the LorentzVector.
    pub fn z(&self) -> f64 {
        self.pz
    }

    /// Returns the energy component of the LorentzVector.
    pub fn e(&self) -> f64 {
        self.e
    }
    /// Returns the x momentum of the LorentzVector.
    pub fn px(&self) -> f64 {
        self.px
    }
    /// Returns the y momentum of the LorentzVector.
    pub fn py(&self) -> f64 {
        self.py
    }
    /// Returns the z momentum of the LorentzVector.
    pub fn pz(&self) -> f64 {
        self.pz
    }

    /// Returns a mutable reference to the time component of the LorentzVector.
    pub fn t_mut(&mut self) -> &mut f64 {
        &mut self.e
    }
    /// Returns a mutable reference to the x component of the LorentzVector.
    pub fn x_mut(&mut self) -> &mut f64 {
        &mut self.px
    }
    /// Returns a mutable reference to the y component of the LorentzVector.
    pub fn y_mut(&mut self) -> &mut f64 {
        &mut self.py
    }
    /// Returns a mutable reference to the z component of the LorentzVector.
    pub fn z_mut(&mut self) -> &mut f64 {
        &mut self.pz
    }

    /// Returns a mutable reference to the energy component of the LorentzVector.
    pub fn e_mut(&mut self) -> &mut f64 {
        &mut self.e
    }
    /// Returns a mutable reference to the x momentum of the LorentzVector.
    pub fn px_mut(&mut self) -> &mut f64 {
        &mut self.px
    }
    /// Returns a mutable reference to the y momentum of the LorentzVector.
    pub fn py_mut(&mut self) -> &mut f64 {
        &mut self.py
    }
    /// Returns a mutable reference to the z momentum of the LorentzVector.
    pub fn pz_mut(&mut self) -> &mut f64 {
        &mut self.pz
    }

    /// Computes the square of the invariant mass of a four-momentum.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[macro_use]
    /// extern crate approx;
    ///
    /// extern crate lorentz_vector;
    /// use lorentz_vector::LorentzVector;
    ///
    /// fn main() {
    ///     let top = LorentzVector::with_mpxpypz(173.2, 0., 103., 300.);
    ///     assert_relative_eq!(top.mass_squared(), 173.2*173.2, max_relative=1e-10);
    /// }
    /// ```
    pub fn mass_squared(&self) -> f64 {
        self.e.powi(2) - self.px.powi(2) - self.py.powi(2) - self.pz.powi(2)
    }

    /// Computes the invariant mass of a four-momentum.
    ///
    /// This function checks that the four-momentum is time- or light-like and will panic
    /// if it is not. This may be a problem for vectors that should be massless, but because of
    /// numerical issues have a tiny negative mass squared. To deal with this case the function
    /// [`mass_root`] exists which emulates the behaviour of the `ROOT` function `M()` which is
    /// safe in this case, but won't panic on any negative mass squared which might not be the
    /// desired behaviour.
    ///
    /// [`mass_root`]: struct.LorentzVector.html#method.mass_root
    ///
    /// # Panics
    ///
    /// If the square of the mass is negative, so if the four-momentum is space like.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[macro_use]
    /// extern crate approx;
    ///
    /// extern crate lorentz_vector;
    /// use lorentz_vector::LorentzVector;
    ///
    /// fn main() {
    ///     let top = LorentzVector::with_mpxpypz(173.2, 0., 103., 300.);
    ///     assert_relative_eq!(top.mass(), 173.2, max_relative=1e-10);
    /// }
    /// ```
    pub fn mass(&self) -> f64 {
        let m_sq = self.mass_squared();
        if m_sq < 0. {
            panic!("LorentzVector::mass: Negative m^2 ({})", m_sq);
        }
        m_sq.sqrt()
    }

    /// Computes the invariant mass of a four-momentum.
    ///
    /// This function emulates the [`M()`] function from `ROOT` when dealing with space-like
    /// four-momenta. If the mass squared of a four-momentum is negative, the returned mass is
    /// computed as `-sqrt(-m^2)`.
    /// See also the function [`mass`] which panics for space-like four-momenta.
    ///
    /// [`M()`]: https://root.cern.ch/doc/master/classTLorentzVector.html#ae3bfd6b5fc3b84aef50f0b0131b906d8
    /// [`mass`]: struct.LorentzVector.html#method.mass
    ///
    /// # Examples
    ///
    /// ```rust
    /// #[macro_use]
    /// extern crate approx;
    ///
    /// extern crate lorentz_vector;
    /// use lorentz_vector::LorentzVector;
    ///
    /// fn main() {
    ///     let top = LorentzVector::with_mpxpypz(173.2, 0., 103., 300.);
    ///     assert_relative_eq!(top.mass_root(), 173.2, max_relative=1e-10);
    ///
    ///     let neg = LorentzVector::with_epxpypz(173.2, 0., 103., 300.);
    ///     assert_relative_eq!(neg.mass_root(), -265.72685223740564, max_relative=1e-10);
    /// }
    /// ```
    pub fn mass_root(&self) -> f64 {
        let mass_sq = self.mass_squared();
        if mass_sq >= 0. {
            mass_sq.sqrt()
        } else {
            -((-mass_sq).sqrt())
        }
    }

    /// Computes the square of the magnitude of the spatial part of
    /// a four-momentum.
    ///
    /// For a four-vector (E, px, py, pz) the returned value is
    /// `px^2 + py^2 + pz^2.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lorentz_vector::LorentzVector;
    ///
    /// let vec = LorentzVector::with_epxpypz(300., 4., -3., 12.);
    /// assert_eq!(vec.mag3_squared(), 13.*13.);
    pub fn mag3_squared(&self) -> f64 {
        self.px.powi(2) + self.py.powi(2) + self.pz.powi(2)
    }

    /// Computes the magnitude of the spatial part of a four-momentum.
    ///
    /// For a four-vector (E, px, py, pz) the returned value is
    /// `sqrt(px^2 + py^2 + pz^2).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lorentz_vector::LorentzVector;
    ///
    /// let vec = LorentzVector::with_epxpypz(300., 4., -3., 12.);
    /// assert_eq!(vec.mag3(), 13.);
    pub fn mag3(&self) -> f64 {
        self.mag3_squared().sqrt()
    }

    /// Computes the square of the transverse momentum of a
    /// four-momentum.
    ///
    /// As usual in high energy physics, the transverse plane is
    /// defined as the x-y-plane, so that p_T^2 = p_x^2 + p_y^2.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lorentz_vector::LorentzVector;
    ///
    /// let vec = LorentzVector::with_epxpypz(200., 4., -3., 20.);
    /// assert_eq!(vec.pt_squared(), 25.);
    /// ```
    pub fn pt_squared(&self) -> f64 {
        self.px.powi(2) + self.py.powi(2)
    }

    /// Computes the transverse momentum of a four-momentum.
    ///
    /// As usual in high energy physics, the transverse plane is
    /// defined as the x-y-plane, so that p_T = sqrt(p_x^2 + p_y^2).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lorentz_vector::LorentzVector;
    ///
    /// let vec = LorentzVector::with_epxpypz(200., 4., -3., 20.);
    /// assert_eq!(vec.pt(), 5.);
    /// ```
    pub fn pt(&self) -> f64 {
        self.pt_squared().sqrt()
    }

    /// Computes the square of the transverse energy of a
    /// four-momentum.
    ///
    /// As usual in high energy physics, the transverse plane is
    /// defined as the x-y-plane, so that E_T^2 = E^2*p_T^2/p^2, where
    /// p is the magnitude of the three-momentum.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lorentz_vector::LorentzVector;
    ///
    /// let vec = LorentzVector::with_epxpypz(200., 4., 0., -3.);
    /// assert_eq!(vec.et_squared(), (200_f64*4./5.).powi(2));
    /// ```
    pub fn et_squared(&self) -> f64 {
        let p_sq = self.mag3_squared();
        if p_sq == 0. {
            0.
        } else {
            self.e.powi(2) * self.pt_squared() / self.mag3_squared()
        }
    }

    /// Computes the transverse energy of a four-momentum.
    ///
    /// As usual in high energy physics, the transverse plane is
    /// defined as the x-y-plane, so that E_T = E*p_T/p, where
    /// p is the magnitude of the three-momentum.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lorentz_vector::LorentzVector;
    ///
    /// let vec = LorentzVector::with_epxpypz(200., 4., 0., -3.);
    /// assert_eq!(vec.et(), 200_f64*4./5.);
    /// ```
    pub fn et(&self) -> f64 {
        self.et_squared().sqrt()
    }
}

impl Default for LorentzVector {
    /// Returns a LorentzVector with all components set to zero.
    fn default() -> LorentzVector {
        LorentzVector::new()
    }
}

impl Add for LorentzVector {
    type Output = LorentzVector;
    fn add(self, rhs: LorentzVector) -> LorentzVector {
        LorentzVector {
            e: self.e + rhs.e,
            px: self.px + rhs.px,
            py: self.py + rhs.py,
            pz: self.pz + rhs.pz,
        }
    }
}
impl<'a> Add<&'a LorentzVector> for LorentzVector {
    type Output = LorentzVector;
    fn add(self, rhs: &'a LorentzVector) -> LorentzVector {
        LorentzVector {
            e: self.e + rhs.e,
            px: self.px + rhs.px,
            py: self.py + rhs.py,
            pz: self.pz + rhs.pz,
        }
    }
}
impl<'a> Add<LorentzVector> for &'a LorentzVector {
    type Output = LorentzVector;
    fn add(self, rhs: LorentzVector) -> LorentzVector {
        LorentzVector {
            e: self.e + rhs.e,
            px: self.px + rhs.px,
            py: self.py + rhs.py,
            pz: self.pz + rhs.pz,
        }
    }
}
impl<'a,'b> Add<&'b LorentzVector> for &'a LorentzVector {
    type Output = LorentzVector;
    fn add(self, rhs: &'b LorentzVector) -> LorentzVector {
        LorentzVector {
            e: self.e + rhs.e,
            px: self.px + rhs.px,
            py: self.py + rhs.py,
            pz: self.pz + rhs.pz,
        }
    }
}
impl AddAssign for LorentzVector {
    fn add_assign(&mut self, rhs: LorentzVector) {
        self.e += rhs.e;
        self.px += rhs.px;
        self.py += rhs.py;
        self.pz += rhs.pz;
    }
}
impl<'a> AddAssign<&'a LorentzVector> for LorentzVector {
    fn add_assign(&mut self, rhs: &'a LorentzVector) {
        self.e += rhs.e;
        self.px += rhs.px;
        self.py += rhs.py;
        self.pz += rhs.pz;
    }
}

impl Sub for LorentzVector {
    type Output = LorentzVector;
    fn sub(self, rhs: LorentzVector) -> LorentzVector {
        LorentzVector {
            e: self.e - rhs.e,
            px: self.px - rhs.px,
            py: self.py - rhs.py,
            pz: self.pz - rhs.pz,
        }
    }
}
impl<'a> Sub<&'a LorentzVector> for LorentzVector {
    type Output = LorentzVector;
    fn sub(self, rhs: &'a LorentzVector) -> LorentzVector {
        LorentzVector {
            e: self.e - rhs.e,
            px: self.px - rhs.px,
            py: self.py - rhs.py,
            pz: self.pz - rhs.pz,
        }
    }
}
impl<'a> Sub<LorentzVector> for &'a LorentzVector {
    type Output = LorentzVector;
    fn sub(self, rhs: LorentzVector) -> LorentzVector {
        LorentzVector {
            e: self.e - rhs.e,
            px: self.px - rhs.px,
            py: self.py - rhs.py,
            pz: self.pz - rhs.pz,
        }
    }
}
impl<'a,'b> Sub<&'b LorentzVector> for &'a LorentzVector {
    type Output = LorentzVector;
    fn sub(self, rhs: &'b LorentzVector) -> LorentzVector {
        LorentzVector {
            e: self.e - rhs.e,
            px: self.px - rhs.px,
            py: self.py - rhs.py,
            pz: self.pz - rhs.pz,
        }
    }
}
impl SubAssign for LorentzVector {
    fn sub_assign(&mut self, rhs: LorentzVector) {
        self.e -= rhs.e;
        self.px -= rhs.px;
        self.py -= rhs.py;
        self.pz -= rhs.pz;
    }
}
impl<'a> SubAssign<&'a LorentzVector> for LorentzVector {
    fn sub_assign(&mut self, rhs: &'a LorentzVector) {
        self.e -= rhs.e;
        self.px -= rhs.px;
        self.py -= rhs.py;
        self.pz -= rhs.pz;
    }
}

impl Mul<f64> for LorentzVector {
    type Output = LorentzVector;
    fn mul(mut self, rhs: f64) -> LorentzVector {
        self.e *= rhs;
        self.px *= rhs;
        self.py *= rhs;
        self.pz *= rhs;
        self
    }
}
impl<'a> Mul<f64> for &'a LorentzVector {
    type Output = LorentzVector;
    fn mul(self, rhs: f64) -> LorentzVector {
        LorentzVector {
            e: self.e * rhs,
            px: self.px * rhs,
            py: self.py * rhs,
            pz: self.pz * rhs,
        }
    }
}
impl<'a> Mul<&'a f64> for LorentzVector {
    type Output = LorentzVector;
    fn mul(self, rhs: &'a f64) -> LorentzVector {
        self * (*rhs)
    }
}
impl<'a, 'b> Mul<&'b f64> for &'a LorentzVector {
    type Output = LorentzVector;
    fn mul(self, rhs: &'b f64) -> LorentzVector {
        self * (*rhs)
    }
}
impl MulAssign<f64> for LorentzVector {
    fn mul_assign(&mut self, rhs: f64) {
        self.e *= rhs;
        self.px *= rhs;
        self.py *= rhs;
        self.pz *= rhs;
    }
}
impl<'a> MulAssign<&'a f64> for LorentzVector {
    fn mul_assign(&mut self, rhs: &'a f64) {
        *self *= *rhs;
    }
}

impl Div<f64> for LorentzVector {
    type Output = LorentzVector;
    fn div(mut self, rhs: f64) -> LorentzVector {
        self.e /= rhs;
        self.px /= rhs;
        self.py /= rhs;
        self.pz /= rhs;
        self
    }
}
impl<'a> Div<f64> for &'a LorentzVector {
    type Output = LorentzVector;
    fn div(self, rhs: f64) -> LorentzVector {
        LorentzVector {
            e: self.e / rhs,
            px: self.px / rhs,
            py: self.py / rhs,
            pz: self.pz / rhs,
        }
    }
}
impl<'a> Div<&'a f64> for LorentzVector {
    type Output = LorentzVector;
    fn div(self, rhs: &'a f64) -> LorentzVector {
        self / (*rhs)
    }
}
impl<'a, 'b> Div<&'b f64> for &'a LorentzVector {
    type Output = LorentzVector;
    fn div(self, rhs: &'b f64) -> LorentzVector {
        self / (*rhs)
    }
}
impl DivAssign<f64> for LorentzVector {
    fn div_assign(&mut self, rhs: f64) {
        self.e /= rhs;
        self.px /= rhs;
        self.py /= rhs;
        self.pz /= rhs;
    }
}
impl<'a> DivAssign<&'a f64> for LorentzVector {
    fn div_assign(&mut self, rhs: &'a f64) {
        *self /= *rhs;
    }
}

impl Index<Component> for LorentzVector {
    type Output = f64;
    fn index(&self, index: Component) -> &f64 {
        match index {
            Component::Zero => &self.e,
            Component::One => &self.px,
            Component::Two => &self.py,
            Component::Three => &self.pz,
        }
    }
}
impl IndexMut<Component> for LorentzVector {
    fn index_mut(&mut self, index: Component) -> &mut f64 {
        match index {
            Component::Zero => &mut self.e,
            Component::One => &mut self.px,
            Component::Two => &mut self.py,
            Component::Three => &mut self.pz,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LorentzVector;

    macro_rules! gen_test {
        ($name:ident => method=$method:ident, expected=$exp:expr, epxpypz=($e:expr, $px:expr, $py:expr, $pz:expr)) => {
            #[test]
            fn $name() {
                let vector = LorentzVector::with_epxpypz($e, $px, $py, $pz);
                let result = vector.$method();
                assert_relative_eq!($exp, result, max_relative=1e-9);
            }
        };
        ($name:ident => method=$method:ident, expected=$exp:expr, mpxpypz=($m:expr, $px:expr, $py:expr, $pz:expr)) => {
            #[test]
            fn $name() {
                let vector = LorentzVector::with_mpxpypz($m, $px, $py, $pz);
                let result = vector.$method();
                assert_relative_eq!($exp, result, max_relative=1e-9);
            }
        };
    }

    mod with_mpxpypz {
        use super::LorentzVector;

        macro_rules! gen_with_mpxpypz_test {
            ($name:ident, $e:expr, $m:expr, $px:expr, $py:expr, $pz:expr) => {
                #[test]
                fn $name() {
                    let rust = LorentzVector::with_mpxpypz($m, $px, $py, $pz);
                    assert_relative_eq!($e, rust.e(), max_relative=1e-10);
                    assert_eq!($px, rust.px());
                    assert_eq!($py, rust.py());
                    assert_eq!($pz, rust.pz());
                }
            }
        }
        gen_with_mpxpypz_test!(zero_vec, 0., 0., 0., 0., 0.);
        gen_with_mpxpypz_test!(zero_mass, 20., 0., 0., 0., 20.);
        gen_with_mpxpypz_test!(m_183, 351.276436786, 183.995576266, -218.434643021, 6.26690336634, -204.420634002);
        gen_with_mpxpypz_test!(m_416, 505.86533915, 416.115278792, -287.365889021, 1.00105672531, -12.9483520745);

        #[test]
        #[should_panic]
        fn negative_mass() {
            LorentzVector::with_mpxpypz(-416.115278792, -287.365889021, 1.00105672531, -12.9483520745);
        }
    }

    mod mass {
        use super::LorentzVector;
        use quickcheck::TestResult;

        gen_test!(zero_vec => method=mass, expected=0., epxpypz=(0., 0., 0., 0.));
        gen_test!(zero_mass => method=mass, expected=0., epxpypz=(30., 0., 0., 30.));
        gen_test!(m_27 => method=mass, expected=27.857051909,
            epxpypz=(484.615676012, 64.1422201998, 477.822408648, 40.593835462));
        gen_test!(m_623 => method=mass, expected=623.259906573,
            epxpypz=(658.149763307, -28.940461633, 43.6343626721, 204.857735697));

        #[test]
        #[should_panic]
        fn negative_mass() {
            let vec = LorentzVector::with_epxpypz(416.115278792, -287.365889021, 1.00105672531, -512.9483520745);
            let m = vec.mass();
            println!("This should have panicked with a negative mass, despite the mass being '{}'", m);
        }

        quickcheck! {
            fn cmp_against_with_mpxpypz(mass: f64, px: f64, py: f64, pz: f64) -> TestResult {
                if mass < 0. {
                    return TestResult::discard();
                }
                let vec = LorentzVector::with_mpxpypz(mass, px, py, pz);
                TestResult::from_bool(relative_eq!(mass, vec.mass(), max_relative=1e-9))
            }
        }
    }

    mod mag3_squared {
        use super::LorentzVector;
        gen_test!(zero_vec => method=mag3_squared, expected=0., epxpypz=(0., 0., 0., 0.));
        gen_test!(only_x => method=mag3_squared, expected=100., epxpypz=(20., 10., 0., 0.));
        gen_test!(only_neg_x => method=mag3_squared, expected=100., epxpypz=(20., -10., 0., 0.));
        gen_test!(only_y => method=mag3_squared, expected=62.*62., epxpypz=(72., 0., 62., 0.));
        gen_test!(only_neg_y => method=mag3_squared, expected=62.*62., epxpypz=(72., 0., -62., 0.));
        gen_test!(only_z => method=mag3_squared, expected=90000., epxpypz=(300., 0., 0., 300.));
        gen_test!(only_neg_z => method=mag3_squared, expected=90000., epxpypz=(800., 0., 0., -300.));
        gen_test!(psq_322758 => method=mag3_squared, expected=322758.194588,
            epxpypz=(569.205225111, -567.158157336, -7.97776293841, 32.033957815));
        gen_test!(psq_304736 => method=mag3_squared, expected=304736.764911,
            epxpypz=(615.906021332, -247.041748803, -491.820038055, -42.6636780736));
    }

    mod mag3 {
        use super::LorentzVector;
        gen_test!(zero_vec => method=mag3, expected=0., epxpypz=(0., 0., 0., 0.));
        gen_test!(only_x => method=mag3, expected=10., epxpypz=(20., 10., 0., 0.));
        gen_test!(only_neg_x => method=mag3, expected=10., epxpypz=(20., -10., 0., 0.));
        gen_test!(only_y => method=mag3, expected=62., epxpypz=(72., 0., 62., 0.));
        gen_test!(only_neg_y => method=mag3, expected=62., epxpypz=(72., 0., -62., 0.));
        gen_test!(only_z => method=mag3, expected=300., epxpypz=(300., 0., 0., 300.));
        gen_test!(only_neg_z => method=mag3, expected=300., epxpypz=(800., 0., 0., -300.));
        gen_test!(p_252 => method=mag3, expected=252.884907845,
            epxpypz=(268.233474351, -30.4568837411, 250.195069847, 20.6296357883));
        gen_test!(p_210 => method=mag3, expected=210.843940223,
            epxpypz=(537.631360064, -5.69712499673, -195.297072922, -79.2575750543));
    }

    mod pt_squared {
        use super::LorentzVector;
        gen_test!(zero_vec => method=pt_squared, expected=0., epxpypz=(0., 0., 0., 0.));
        gen_test!(zero_pt => method=pt_squared, expected=0., epxpypz=(40., 0., 0., 40.));
        gen_test!(only_x => method=pt_squared, expected=100., epxpypz=(10., 10., 0., 0.));
        gen_test!(only_neg_x => method=pt_squared, expected=100., epxpypz=(10., -10., 0., 0.));
        gen_test!(only_y => method=pt_squared, expected=62.*62., epxpypz=(62., 0., 62., 0.));
        gen_test!(only_neg_y => method=pt_squared, expected=62.*62., epxpypz=(62., 0., -62., 0.));
        gen_test!(pt_315 => method=pt_squared, expected=315.213245164_f64.powi(2),
            epxpypz=(369.671172305, -56.1746860374, -310.167365426, 192.968794135));
        gen_test!(pt_119 => method=pt_squared, expected=119.605738865_f64.powi(2),
            epxpypz=(1716.31673646, -81.4393539327, -87.596600394, 251.364268921));

    }

    mod pt {
        use super::LorentzVector;
        gen_test!(zero_vec   => method=pt, expected=0., epxpypz=(0., 0., 0., 0.));
        gen_test!(zero_pt    => method=pt, expected=0., epxpypz=(40., 0., 0., 40.));
        gen_test!(only_x     => method=pt, expected=10., epxpypz=(10., 10., 0., 0.));
        gen_test!(only_neg_x => method=pt, expected=10., epxpypz=(10., -10., 0., 0.));
        gen_test!(only_y     => method=pt, expected=62., epxpypz=(62., 0., 62., 0.));
        gen_test!(only_neg_y => method=pt, expected=62., epxpypz=(62., 0., -62., 0.));
        gen_test!(pt_144     => method=pt, expected=144.911221504,
            epxpypz=(155.40127331, 15.3726963439, 144.093519372, 43.24658756));
        gen_test!(pt_231     => method=pt, expected=231.989521619,
            epxpypz=(785.226886448, 217.637082227, -80.3320520148, -100.00571379));
    }

    mod et_squared {
        use super::LorentzVector;
        gen_test!(zero_vec => method=et_squared, expected=0., epxpypz=(0., 0., 0., 0.));
        gen_test!(zero_et_massless => method=et_squared, expected=0., epxpypz=(300., 0., 0., -300.));
        gen_test!(zero_et_massive => method=et_squared, expected=0., epxpypz=(800., 0., 0., -300.));
        gen_test!(only_x => method=et_squared, expected=400., epxpypz=(20., 10., 0., 0.));
        gen_test!(only_neg_x => method=et_squared, expected=400., epxpypz=(20., -10., 0., 0.));
        gen_test!(only_y => method=et_squared, expected=72.*72., epxpypz=(72., 0., 62., 0.));
        gen_test!(only_neg_y => method=et_squared, expected=72.*72., epxpypz=(72., 0., -62., 0.));
        gen_test!(et_347 => method=et_squared, expected=347.924589557_f64.powi(2),
            epxpypz=(400.616585689, -346.739438367, -18.3753575611, -198.201657029));
        gen_test!(et_305 => method=et_squared, expected=305.620406211_f64.powi(2),
            epxpypz=(317.307612993, -26.2268883955, 300.241479261, 84.1420887216));

    }

    mod et {
        use super::LorentzVector;
        gen_test!(zero_vec => method=et, expected=0., epxpypz=(0., 0., 0., 0.));
        gen_test!(zero_et_massless => method=et, expected=0., epxpypz=(300., 0., 0., -300.));
        gen_test!(zero_et_massive => method=et, expected=0., epxpypz=(800., 0., 0., -300.));
        gen_test!(only_x => method=et, expected=20., epxpypz=(20., 10., 0., 0.));
        gen_test!(only_neg_x => method=et, expected=20., epxpypz=(20., -10., 0., 0.));
        gen_test!(only_y => method=et, expected=72., epxpypz=(72., 0., 62., 0.));
        gen_test!(only_neg_y => method=et, expected=72., epxpypz=(72., 0., -62., 0.));
        gen_test!(et_158 => method=et, expected=158.230719208,
            epxpypz=(170.837911448, 145.648264664, 32.1009358759, 60.7109415654));
        gen_test!(et_216 => method=et, expected=216.129121105,
            epxpypz=(359.054975674, -197.767284995, -87.1345040568, -286.697612733));
    }

    #[test]
    fn it_works() {
    }
}
