perms_dict = {
    3: [
        (1,3,2)
    ],
    4: [
        (1,3,2),
        (1,4,2),
        (1,3,4),
        (2,3,4),
    ],
    5: [
        (1,3,2),
        (1,4,2),
        (1,5,2),
        (1,3,4),
        (1,3,5),
        (1,4,5),
        (2,3,4),
        (2,3,5),
        (2,4,5),
        (3,4,5),
    ]
}

def getPermutations(n):
    return perms_dict[n]