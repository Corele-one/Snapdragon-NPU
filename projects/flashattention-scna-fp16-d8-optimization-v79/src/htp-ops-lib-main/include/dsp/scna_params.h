#pragma once

/*
 * Frozen SCOPE Exp2 d8 deployment parameters.
 * Source checkpoint SHA-256:
 * 01ed8dadb29f03da0031d236ebff4e3c51f8d2339e1713b4ff5b2a96a34e9bd7
 * Domain: [-256, 0]. All coefficients are non-negative after fusing the
 * one-hidden-layer output weights into the affine terms.
 */

#define SCNA_MIN_INPUT (-256.0f)
#define SCNA_MAX_INPUT (0.0f)
#define SCNA_D8_WIDTH 8

static const __fp16 scna_exp2_d8_wk[SCNA_D8_WIDTH] = {
  (__fp16) 2.586841583251953e-05f,
  (__fp16) 0.00010031461715698242f,
  (__fp16) 0.0004892349243164062f,
  (__fp16) 0.002384185791015625f,
  (__fp16) 0.011627197265625f,
  (__fp16) 0.05670166015625f,
  (__fp16) 0.2763671875f,
  (__fp16) 0.345458984375f,
};

static const __fp16 scna_exp2_d8_bk[SCNA_D8_WIDTH] = {
  (__fp16) 0.0004138946533203125f,
  (__fp16) 0.0013751983642578125f,
  (__fp16) 0.005588531494140625f,
  (__fp16) 0.0218048095703125f,
  (__fp16) 0.0797119140625f,
  (__fp16) 0.25927734375f,
  (__fp16) 0.6318359375f,
  (__fp16) 0.0f,
};

/* Algebraically equivalent piecewise-linear form of the seven active d8
 * neurons.  Neuron 7 is ReLU(w*x) with w>0 and x in [-256,0], hence is
 * identically zero on the deployed domain.  Prefix coefficients are rounded
 * to FP16 intentionally; the resulting reassociation is accuracy-gated. */
#define SCNA_D8_ACTIVE_WIDTH 7
static const __fp16 scna_exp2_d8_breakpoint[SCNA_D8_ACTIVE_WIDTH] = {
  (__fp16) -16.0f,
  (__fp16) -13.7109375f,
  (__fp16) -11.421875f,
  (__fp16) -9.1484375f,
  (__fp16) -6.85546875f,
  (__fp16) -4.57421875f,
  (__fp16) -2.287109375f,
};
static const __fp16 scna_exp2_d8_prefix_w[SCNA_D8_ACTIVE_WIDTH] = {
  (__fp16) 2.586841583251953e-05f,
  (__fp16) 0.00012612342834472656f,
  (__fp16) 0.0006155967712402344f,
  (__fp16) 0.0030002593994140625f,
  (__fp16) 0.01462554931640625f,
  (__fp16) 0.07135009765625f,
  (__fp16) 0.34765625f,
};
static const __fp16 scna_exp2_d8_prefix_b[SCNA_D8_ACTIVE_WIDTH] = {
  (__fp16) 0.0004138946533203125f,
  (__fp16) 0.001789093017578125f,
  (__fp16) 0.00737762451171875f,
  (__fp16) 0.0291748046875f,
  (__fp16) 0.10888671875f,
  (__fp16) 0.3681640625f,
  (__fp16) 1.0f,
};
