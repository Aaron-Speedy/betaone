#!/bin/sh

set -xe

CC="${CXX:-cc}"

gcc beta.c -o beta -Wall -pedantic -ggdb -O3 -std=c11 -lraylib
