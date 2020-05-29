import inspect


def _call_with_is_training_flag_if_possible(model, inputs, is_training):
  argspec = inspect.getfullargspec(model)
  if 'is_training' in argspec.args or argspec.varkw is not None:
    outputs = model(inputs, is_training=is_training)
  else:
    outputs = model(inputs)
  return outputs