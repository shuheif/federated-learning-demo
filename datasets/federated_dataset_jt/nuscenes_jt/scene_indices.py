SCENE_INDICES_IGNORED = [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312,
                         313,
                         314, 419]  # 419 is not in canbus blacklist but it's speed is not okay

SCENE_INDICES_BOS_TRAIN = [287, 1091, 512, 892, 705, 305, 158, 802, 415, 1065, 813, 912, 907, 560, 751, 231, 546, 332,
                           441, 1052,
                           256, 180,
                           632, 364, 904, 586, 368, 1077, 792, 509, 761, 1051, 1078, 371, 139, 379, 1090, 634, 765,
                           1064, 1014, 641,
                           733, 356, 28,
                           716, 931, 177, 286, 769, 414, 227, 257, 629, 1017, 1066, 1016, 803, 678, 72, 915, 562, 917,
                           1108, 13,
                           187, 56, 975, 646,
                           757, 465, 5, 232, 905, 382, 128, 428, 43, 955, 519, 422, 923, 1057, 901, 717, 949, 361, 1019,
                           529, 246,
                           474, 208, 159,
                           433, 394, 577, 125, 1062, 260, 377, 1003, 658, 253, 400, 444, 947, 871, 650, 746, 1076, 764,
                           44, 65, 462,
                           1011, 638, 957,
                           853, 391, 295, 740, 330, 962, 29, 447, 531, 427, 925, 924, 1082, 1097, 479, 138, 900, 1099,
                           711, 75, 237,
                           972, 676, 181,
                           202, 997, 984, 961, 561, 230, 385, 880, 124, 179, 668, 735, 1020, 62, 1109, 663, 194, 437,
                           644, 222,
                           1110, 1071, 242, 1056,
                           359, 463, 967, 656, 104, 191, 98, 383, 351, 876, 778, 24, 718, 106, 643, 53, 645, 500, 452,
                           920, 777, 46,
                           304, 599, 981,
                           123, 149, 270, 994, 821, 804, 908, 726, 220, 945, 787, 258, 247, 451, 927, 597, 762, 213,
                           510, 283, 636,
                           976, 501, 33, 928,
                           666, 796, 662, 1079, 783, 303, 664, 370, 869, 1089, 571, 69, 155, 808, 1018, 593, 815, 1058,
                           374, 895,
                           373, 667, 35, 36, 775,
                           914, 392, 469, 996, 245, 417, 294, 583, 390, 297, 534, 254, 882, 74, 11, 26, 218, 865, 600,
                           539, 132,
                           558, 655, 302, 67, 448,
                           193, 669, 505, 477, 1046, 866, 517, 108, 640, 868, 318, 190, 982, 805, 1087, 21, 49, 93, 226,
                           708, 809,
                           131, 1086, 1107, 19,
                           468, 475, 426, 51, 275, 737, 405, 899, 1059, 653, 1047, 584, 157, 1025, 456, 323, 719, 7, 73,
                           893, 710,
                           536, 582, 130, 681,
                           1045, 476, 1084, 34, 263, 464, 436, 133, 450, 806, 57, 873, 416, 851, 872, 1070, 979, 252,
                           76, 822, 862,
                           403, 991, 1050, 713,
                           60, 278, 630, 1049, 352, 649, 983, 884, 684, 596, 633, 626, 151, 515, 758, 543, 324, 730,
                           855, 556, 695,
                           12, 63, 306, 522,
                           389, 375, 1012, 17, 1100, 697, 1022, 922, 789, 467, 195, 103, 1008, 276, 184, 563, 262, 150,
                           188, 1060,
                           381, 134, 763, 443,
                           243, 728, 478, 812, 55, 360, 411, 749, 236, 182, 966, 902, 395, 588, 661, 25, 54, 110, 502,
                           449, 209,
                           207, 738, 59, 1021, 911,
                           771, 398, 741, 408, 200, 122, 784, 525, 752, 212, 160, 1094, 192, 71, 272, 856, 675, 1080,
                           228, 576, 598,
                           759, 299, 523, 206,
                           894, 674, 41, 909, 240, 511, 204, 251, 438, 358, 380, 1068, 689, 528, 353, 677, 9, 864, 97,
                           101, 679,
                           432, 1044, 513, 552, 225,
                           203, 850, 570, 916, 672, 434, 506, 249, 878, 393, 454, 861, 1088, 589, 978, 397, 269, 910,
                           526, 2, 544,
                           889, 446, 627, 648, 760,
                           50, 424, 407, 121, 820, 557, 290, 953, 183, 715, 129, 4, 988, 52, 264, 860, 652, 15, 367,
                           350, 566, 568,
                           791, 1095, 688, 642,
                           1006, 647, 1096, 440, 401, 268, 8, 480, 95, 421, 107, 459, 977, 559, 3, 659, 39, 16, 301,
                           439, 706, 535,
                           406, 292, 261, 399,
                           1005, 1072, 241, 651, 386, 1098, 959, 1069, 1007, 508, 592, 248, 956, 274, 328, 100, 1009,
                           701, 66, 565,
                           657, 127, 575, 906,
                           794, 402, 384, 27, 354, 344, 709, 105, 696, 277, 594, 1105, 877, 896, 554, 239, 499, 887,
                           1048, 1083,
                           541, 698, 58, 995, 714,
                           635, 590, 238, 457, 574, 285]

SCENE_INDICES_SGP_VAL = [30, 38, 126, 259, 291, 298, 316, 362, 365, 376, 378, 410, 429, 471, 504, 507, 524, 527, 542,
                         770, 781, 797,
                         810, 819, 852,
                         863, 883, 885, 921, 960, 998, 1024, 1055, 1067]

SCENE_INDICES_SGP_TEST = [847, 816, 321, 135, 1010, 712, 458, 288, 418, 1004, 425, 20, 514, 244, 372, 671, 886, 154,
                          455, 229, 357,
                          870, 795, 1,
                          22, 747, 929, 396, 273, 70, 315, 1106, 369, 1001, 1063, 1000, 739, 990, 811, 18, 707, 210,
                          199, 572, 665,
                          858, 1073, 234, 798,
                          214, 800, 152, 45, 363, 744, 578, 293, 219, 96, 686, 790, 300, 731, 1053, 1093, 329, 413,
                          1054, 120, 848,
                          537, 530, 453, 958,
                          521, 178, 14, 431, 897, 683, 786, 538, 68, 255, 585, 587, 1023, 109, 553, 999, 700, 854, 42,
                          780, 591,
                          625, 532, 250, 564, 48,
                          430, 461, 670, 366, 61, 963, 573, 703, 64, 969, 930, 1102, 211, 442, 271, 518, 639, 926, 1013,
                          545, 595,
                          768, 734, 980, 47,
                          891, 99, 1075, 345, 445, 388, 952, 533, 660, 704, 555, 185, 1061, 968, 1015, 849, 971, 348,
                          888, 1081,
                          102, 233, 898, 435, 992,
                          420, 1104, 913, 196, 355, 23, 349, 346, 654, 1002, 919, 875, 317, 1085, 817, 224, 750, 673,
                          296, 412, 221,
                          1074, 92, 6, 1101,
                          284, 289, 235, 32, 736, 782, 989, 347, 94, 520, 423, 472, 331, 727, 767, 903, 580, 10, 31,
                          1092, 637, 687,
                          890, 685, 799]

SCENE_INDICES_VALID = SCENE_INDICES_SGP_TEST + SCENE_INDICES_SGP_VAL + SCENE_INDICES_BOS_TRAIN