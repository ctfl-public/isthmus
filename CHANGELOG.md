# Changelog

## Unreleased

### Added
- New feature Y.

## X.X.X - 2025-07-30

### tests/test_verification_cases.py
- Lots of new testing functions, basically just the 2D versions of voxel division, surface creation, and voxel association tests
- New tests added to the TestCase classes
- Small changes for clarification and 2D compatibility

### src/isthmus_prototype.py
- New classes, many/all of which are 2D versions of existing classes
- Adding the 'ndims' variable to MC_System class to set 2D/3D
- Argument validity checks modified to be 2D compatible
- The initializer of MC_System (the parent function of isthmus) has been modified to be 2D-compatible; some function calls are shared between 2D and 3D, but others differ and are separated by a conditional which checks 'ndims'
- New functions for 2D
- The member functions get_element() and get_indices() in Grid class have been changed to take in lists of x,y,z indices rather than individually
- The s_voxel_ids member variable in the Triangle class has been removed for redundancy
- A 'noise' function that I use to slightly change floats when testing. As an extra check to running the normal tests, I ran the 2D tests with about 1-5% noise added to relevant quantities to see if the test would fail. Basically make sure the tests don't succeed because I wrote them badly but because they actually test the desired behavior.

### Added
- Initial release with feature X.

### Fixed
- Resolved issue Z.
