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

## Examples

See [tests/GoldenTests.cpp](tests/GoldenTests.cpp) for an example of how the golden library is used.

See [tests/ProtobufSilver.cpp](tests/ProtobufSilver.cpp) for an example of how the silver sub-library is used.

# Silver

Silver is an analogue to Golden except instead of testing against a value it stores a value to
used as input for a test later.

It is probably best to give a use case example.

## Use Case

Say you have a function that is called by another function during the run of the program.
Suppose that function relies on inputs (potentially local, global, etc.) from earlier in the
program. If it is legacy code, you may not know what inputs to said function are valid inputs
for a unit test; however, we still want to unit test. You can use Silver to record the inputs
to that function during a program run. Afterwards, this input becomes your "silver".

When we want to unit test the function, we can "desilver" the silver into the program state.
Now, we can call the function as it has been called previously.

In combination with golden, we can record a functions input and output for a program run at some
set version. Then as we refactor, etc. we can test our functions with the "silver" input
against the "gold" we have previously set.