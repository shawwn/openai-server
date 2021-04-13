import tensorflow.compat.v1 as tf

import model


def lerp(a, b, t):
    return (b - a) * t + a


# OOMs for 1.5B
def penalize_used_expensive(logits, output, frequency_penalty=0.85):
    penalty = tf.reduce_min(tf.one_hot(output, logits.shape[1], frequency_penalty, 1.0), axis=1)
    minimum_logit = tf.reduce_min(logits, axis=1)
    change_tensor = minimum_logit * tf.ones_like(logits, dtype=logits.dtype)
    result = lerp(change_tensor, logits, penalty)
    return result


# Doesn't seem quite right
def penalize_used_new(logits, output, frequency_penalty=0.85):
    # I want to change the indices of logits wherever the index is found in output
    minimum_logit = tf.reduce_min(logits, axis=1)
    unique = tf.unique(output[0])[0]
    ones = tf.ones_like(unique, dtype=unique.dtype)
    indices = tf.expand_dims(unique, 1)

    updates = tf.scatter_nd(indices, ones, [logits.shape[1]])

    bool_tensor = tf.expand_dims(tf.cast(updates, tf.bool), 0)

    result = tf.compat.v1.where(bool_tensor, lerp(minimum_logit, logits, frequency_penalty), logits)
    return result


# Only works for 1558M
def penalize_used(logits, output, frequency_penalty=0.85):
    # I want to change the indices of logits wherever the index is found in output
    unique = tf.unique(output[0])[0]
    ones = tf.ones_like(unique, dtype=unique.dtype)
    indices = tf.expand_dims(unique, 1)

    updates = tf.scatter_nd(indices, ones, [logits.shape[1]])

    bool_tensor = tf.expand_dims(tf.cast(updates, tf.bool), 0)

    result = tf.compat.v1.where(bool_tensor, logits * frequency_penalty, logits)
    return result


def top_k_logits(logits, k, epsilon=-1e10):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * epsilon,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p, epsilon=1e10):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * epsilon,
        logits,
    )


# def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1, frequency_penalty=0.0):
#     if start_token is None:
#         assert context is not None, 'Specify exactly one of start_token and context!'
#     else:
#         assert context is None, 'Specify exactly one of start_token and context!'
#         context = tf.fill([batch_size, 1], start_token)

#     def step(hparams, tokens, past=None):
#         lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

#         logits = lm_output['logits'][:, :, :hparams.n_vocab]
#         presents = lm_output['present']
#         presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
#         return {
#             'logits': logits,
#             'presents': presents,
#         }

#     with tf.name_scope('sample_sequence'):
#         def body(past, prev, output):
#           prev2 = prev[:, tf.newaxis]
#           # import pdb; pdb.set_trace()
#           next_outputs = step(hparams, prev2, past=past)
#           logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
#           # TODO: This is causing problems for models smaller than
#           # 1558M, so disable it for now.
#           # if frequency_penalty != 0.0:
#           #     logits = penalize_used(logits, output, frequency_penalty=frequency_penalty)
#           # logits = top_k_logits(logits, k=top_k)
#           # logits = top_p_logits(logits, p=top_p)
#           samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
#           # import pdb; pdb.set_trace()
#           return (
#               next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
#               tf.squeeze(samples, axis=[1]),
#               tf.concat([output, samples], axis=1)
#           )

#         prev = context[:, -1]
#         output = context

#         #import pdb; pdb.set_trace()
#         if True:
#           maximum_iterations = length - 1
#           past, prev, output = body(None, prev, output)
#         else:
#           maximum_iterations = length
#           sh=model.past_shape(hparams=hparams, batch_size=batch_size);
#           sh[-2] = 0;
#           past = tf.zeros(sh)

#         def cond(*args):
#             return True

#         _, _, tokens = tf.while_loop(
#             cond=cond, body=body,
#             maximum_iterations=length,
#             loop_vars=[
#                 past,
#                 prev,
#                 output,
#             ],
#             shape_invariants=[
#                 tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
#                 tf.TensorShape([batch_size]),
#                 tf.TensorShape([batch_size, None]),
#             ],
#             back_prop=False,
#         )

#         return tokens


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0, epsilon=-1e10, frequency_penalty=0.0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
        if hparams.dtype != tf.float32:
            lm_output["logits"] = tf.cast(lm_output["logits"], tf.float32)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output):
            # import pdb; pdb.set_trace()
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            # if frequency_penalty != 0.0:
            # logits = penalize_used(logits, output, penalize=frequency_penalty)
            # if top_p != 0.0:
            # logits = top_p_logits(logits, p=top_p, epsilon=epsilon)
            # else:
            #logits = top_k_logits(logits, k=top_k, epsilon=epsilon)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens

