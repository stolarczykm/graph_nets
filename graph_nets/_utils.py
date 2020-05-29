import inspect


def _has_argument(callable_, argument_name):
  return argument_name in inspect.getfullargspec(callable_).args


def _call_with_is_training_flag_if_possible(model, inputs, is_training):
  if _has_argument(model, 'is_training'):
    assert is_training is not None
    outputs = model(inputs, is_training=is_training)
  else:
    outputs = model(inputs)
  return outputs