# Golden

Golden is a test framework for doing approval tests in the vein of
https://github.com/approvals/ApprovalTests.cpp.

Instead of operating on pure text, golden uses a serializer to save data
to and from a file. This data is read from the file and compared against
the current data for differences.

## Why not pure text

Pure text can be good for some scenarios such as UI testing (including some floating-point
UI testing with specified rounding); however, this becomes difficult to use if you are
working with scientific data with lots of floating-point entries, etc.

You could write an array of floats to a pure text file with a precision of 10 decimal places;
however, this would be inefficient.

Instead, we compare pure data to pure data. Currently, we do this using Protobuf, but
our system should be extendable in the future to other serialization libraries. The nice
thing about Protobuf is that it has facilities for comparing two messages for equality,
including handling floating-point error.