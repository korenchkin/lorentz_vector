# LorentzVector

A library for working with four-vectors (or Lorentz vectors).

This library is mostly aimed at high energy physics applications,
but should be general enough for usage in special relativity.

## Design

The library is intended to be usable for both space-time vectors and
four-momenta.  Therefore this library includes functions and methods
with the common names for both usecases.

The design of this library is strongly inspired by the class
[`TLorentzVector`] from the C++ particle physics data analysis
framework [ROOT]. However it differs from `TLorentzVector in several
key aspects which are listed in the [Comparison with TLorentzVector]
section.

[`TLorentzVector`]: https://root.cern.ch/doc/master/classTLorentzVector.html
[ROOT]: https://root.cern.ch

## Examples

```rust
#[macro_use]
extern crate approx;

extern crate lorentz_vector;
use lorentz_vector::LorentzVector;

fn main() {
    let top = LorentzVector::with_mpxpypz(173.2, 0., 103., 300.);
    assert_relative_eq!(top.mass(), 173.2, max_relative=1e-10);
}
```
