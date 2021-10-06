import numpy as np


class PBCArray(np.ndarray):
    """
    A ndarray with periodic boundary conditions when slicing (a.k.a. wrap).

    Any rank is supported.

    Examples
    --------
    2D array for semplicity of visualization, any rank should work.

    >>> dim = 3
    >>> A = np.zeros((dim,dim),dtype=int)
    >>> for i in range(dim):
    ...     A[i,i] = i+1

    >>> A = PBCArray(A)
    >>> print(A)
    [[1 0 0]
     [0 2 0]
     [0 0 3]]

    >>> print(A[-dim:,:2*dim])
    [[1 0 0 1 0 0]
     [0 2 0 0 2 0]
     [0 0 3 0 0 3]
     [1 0 0 1 0 0]
     [0 2 0 0 2 0]
     [0 0 3 0 0 3]]

    """

    def __new__(cls, pos):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(pos).view(cls)
        # add the new attribute to the created instance
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

    def __getitem__(self, index):
        """
        Completely general method, works with integers, slices and ellipses,
        Periodic boundary conditions are taken into account by rolling and
        padding the array along the dimensions that need it.
        Slices with negative indexes need special treatment.

        """
        shape_ = self.shape
        rank = len(shape_)

        slices = _reconstruct_full_slices(shape_, index)
        # Now actually slice with pbc along each direction.
        newarr = self
        slice_tup = [slice(None)] * rank

        for idim, sli in enumerate(slices):
            if isinstance(sli, int):
                slice_tup[idim] = sli % shape_[idim]
            elif isinstance(sli, slice):
                roll, pad, start, stop, step = _check_slice(sli, shape_[idim])
                # If the beginning of the slice does not coincide with a grid point
                # equivalent to 0, roll the array along that axis until it does
                if roll > 0:
                    newarr = np.roll(newarr, roll, axis=idim)
                # If the span of the slice extends beyond the boundaries of the array,
                # pad the array along that axis until we have enough elements.
                if pad > 0:
                    pad_tup = [(0, 0)] * rank
                    pad_tup[idim] = (0, pad)
                    newarr = np.pad(newarr, pad_tup, mode="wrap")
                slice_tup[idim] = slice(start, stop, step)

        slice_tup = tuple(slice_tup)

        return np.ndarray.__getitem__(newarr, slice_tup)


def _reconstruct_full_slices(shape_, index):
    """
    Auxiliary function for __getitem__ to reconstruct the explicit slicing
    of the array if there are ellipsis or missing axes.

    """
    if not isinstance(index, tuple):
        index = (index,)
    slices = []
    idx_len, rank = len(index), len(shape_)

    for slice_ in index:
        if slice_ is Ellipsis:
            slices.extend([slice(None)] * (rank + 1 - idx_len))
        elif isinstance(slice_, slice):
            slices.append(slice_)
        elif isinstance(slice_, (int)):
            slices.append(slice_)

    sli_len = len(slices)
    if sli_len > rank:
        msg = "too many indices for array"
        raise IndexError(msg)
    elif sli_len < rank:
        slices.extend([slice(None)] * (rank - sli_len))

    return slices


def _order_slices(dim, slices, shape_):
    """
    Order the slices span in ascending order.
    When we are slicing a pbcarray we might be rolling and padding the array
    so it's probably a good idea to make the array as small as possible
    early on.

    """
    sizes = []
    for idim, sli in slices:
        step = sli.step or 1
        start = sli.start or (0 if step > 0 else shape_[idim])
        stop = sli.stop or (shape_[idim] if step > 0 else 0)
        size = abs((max(start, stop) - min(start, stop)) // step)
        sizes.append(size)

    sizes, slices = zip(*sorted(zip(sizes, slices)))

    return slices


def _check_slice(sli, dim):
    """
    Check if the current slice needs to be treated with pbc or if we can
    simply pass it to ndarray __getitem__.

    Slice is special in the following cases:
    if sli.start < 0 or > dim           # roll (and possibly pad)
    if sli.stop > dim or < 0            # roll (and possibly pad)
    if abs(sli.stop - sli.start) > 0    # pad
    """
    _roll = 0
    _pad = 0

    step = sli.step or 1
    start = (0 if step > 0 else dim) if sli.start is None else sli.start
    stop = (dim if step > 0 else 0) if sli.stop is None else sli.stop
    span = stop - start if step > 0 else start - stop

    if span <= 0:
        return _roll, _pad, sli.start, sli.stop, sli.step

    lower = min(start, stop)
    upper = max(start, stop)
    _start = 0 if step > 0 else span
    _stop = span if step > 0 else 0
    if span > dim:
        _pad = span - dim
        _roll = -lower % dim
    elif lower < 0 or upper > dim:
        _roll = -lower % dim
    else:
        _start = sli.start
        _stop = sli.stop

    return _roll, _pad, _start, _stop, step