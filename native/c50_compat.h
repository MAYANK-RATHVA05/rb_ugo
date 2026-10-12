/* SPDX-License-Identifier: GPL-3.0-or-later
 * Minimal process-local compatibility declarations for CRAN C50's C bridge.
 * No R runtime or R random-number generator is emulated.
 */
#ifndef UGO_C50_COMPAT_H
#define UGO_C50_COMPAT_H
#include <stddef.h>
void Rprintf(const char *format, ...);
void error(const char *format, ...);
void *R_alloc(size_t count, int size);
void GetRNGstate(void);
void PutRNGstate(void);
double runif(double a, double b);
#endif
