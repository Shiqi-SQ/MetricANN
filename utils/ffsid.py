DEFAULT_TABLE = "fzGBVhoNcZUC3vXwgDWkEnumpy7LiRaj6SJF1r8xqY2s4KTMQ9A5HPbted"
_POS = [9, 8, 1, 6, 2, 4]
_XOR = 177451812
_ADD = 8728348608

def ffsid_encode(id_num: int, table: str = DEFAULT_TABLE) -> str:
    if id_num > 29460791295 or id_num < -8858370048:
        raise ValueError("ID out of range")
    num = (id_num ^ _XOR) + _ADD
    r = list("1  4 1 7  ")
    for i in range(6):
        r[_POS[i]] = table[num // (58 ** i) % 58]
    return "".join(r)

def ffsid_decode(ffsid: str, table: str = DEFAULT_TABLE) -> int:
    tr = {c: i for i, c in enumerate(table)}
    r = 0
    for i in range(6):
        r += tr[ffsid[_POS[i]]] * (58 ** i)
    return (r - _ADD) ^ _XOR
