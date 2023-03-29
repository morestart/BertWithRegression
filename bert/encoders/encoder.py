import torch


class Encoder(object):
    """
    Base class for a encoder employing an identity function.

    Args:
        enforce_reversible (bool, optional): Check for reversibility on ``Encoder.encode`` and
          ``Encoder.decode``. Formally, reversible means:
          ``Encoder.decode(Encoder.encode(object_)) == object_``.
    """

    def __init__(self, enforce_reversible=False):
        self.enforce_reversible = enforce_reversible

    def encode(self, object_):
        """ Encodes an object.

        Args:
            object_ (object): Object to encode.

        Returns:
            object: Encoding of the object.
        """
        if self.enforce_reversible:
            self.enforce_reversible = False
            encoded_decoded = self.decode(self.encode(object_))
            self.enforce_reversible = True
            if encoded_decoded != object_:
                raise ValueError('Encoding is not reversible for "%s"' % object_)

        return object_

    def batch_encode(self, iterator, *args, **kwargs):
        """
        Args:
            batch (list): Batch of objects to encode.
            *args: Arguments passed to ``encode``.
            **kwargs: Keyword arguments passed to ``encode``.

        Returns:
            list: Batch of encoded objects.
        """
        return [self.encode(object_, *args, **kwargs) for object_ in iterator]

    def decode(self, encoded):
        """ Decodes an object.

        Args:
            object_ (object): Encoded object.

        Returns:
            object: Object decoded.
        """
        if self.enforce_reversible:
            self.enforce_reversible = False
            decoded_encoded = self.encode(self.decode(encoded))
            self.enforce_reversible = True
            if decoded_encoded != encoded:
                raise ValueError('Decoding is not reversible for "%s"' % encoded)

        return encoded

    def batch_decode(self, iterator, *args, **kwargs):
        """
        Args:
            iterator (list): Batch of encoded objects.
            *args: Arguments passed to ``decode``.
            **kwargs: Keyword arguments passed to ``decode``.

        Returns:
            list: Batch of decoded objects.
        """
        return [self.decode(encoded, *args, **kwargs) for encoded in iterator]


def is_namedtuple(object_):
    return hasattr(object_, '_asdict') and isinstance(object_, tuple)


def collate_tensors(batch, stack_tensors=torch.stack):
    """ Collate a list of type ``k`` (dict, namedtuple, list, etc.) with tensors.

    Inspired by:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L31

    Args:
        batch (list of k): List of rows of type ``k``.
        stack_tensors (callable): Function to stack tensors into a batch.

    Returns:
        k: Collated batch of type ``k``.

    Example use case:
        This is useful with ``torch.utils.data.dataloader.DataLoader`` which requires a collate
        function. Typically, when collating sequences you'd set
        ``collate_fn=partial(collate_tensors, stack_tensors=encoders.text.stack_and_pad_tensors)``.

    Example:

        >>> import torch
        >>> batch = [
        ...   { 'column_a': torch.randn(5), 'column_b': torch.randn(5) },
        ...   { 'column_a': torch.randn(5), 'column_b': torch.randn(5) },
        ... ]
        >>> collated = collate_tensors(batch)
        >>> {k: t.size() for (k, t) in collated.items()}
        {'column_a': torch.Size([2, 5]), 'column_b': torch.Size([2, 5])}
    """
    if all([torch.is_tensor(b) for b in batch]):
        return stack_tensors(batch)
    if (all([isinstance(b, dict) for b in batch]) and
            all([b.keys() == batch[0].keys() for b in batch])):
        return {key: collate_tensors([d[key] for d in batch], stack_tensors) for key in batch[0]}
    elif all([is_namedtuple(b) for b in batch]):  # Handle ``namedtuple``
        return batch[0].__class__(**collate_tensors([b._asdict() for b in batch], stack_tensors))
    elif all([isinstance(b, list) for b in batch]):
        # Handle list of lists such each list has some column to be batched, similar to:
        # [['a', 'b'], ['a', 'b']] → [['a', 'a'], ['b', 'b']]
        transposed = zip(*batch)
        return [collate_tensors(samples, stack_tensors) for samples in transposed]
    else:
        return batch
