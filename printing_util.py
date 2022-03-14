# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation



# returns True when a number is of the form X0000 where X={1,2,... 9}.
# this is useful for printing messages with a logarithmic frequency where
# the number may siginificantly vary.
# for a short run, we get a detailed information,
# while for a long run, we get a less frequent information.

import math
def number_has_leading_zeros_p(n):
    if n == 0:
        return True
    ten_exponential = 10**math.floor(math.log10(n))
    return (n % ten_exponential) == 0
