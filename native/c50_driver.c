/* SPDX-License-Identifier: GPL-3.0-or-later
 * Process-isolated interface to the original CRAN C50 learner and predictor.
 * Upstream learner sources, attribution and GPL headers are retained by setup.
 */
#include "c50_compat.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ugo_c50(char **, char **, char **, int *, int *, int *, int *, int *,
             double *, int *, int *, double *, int *, int *, int *,
             char **, char **, char **);
void ugo_predictions(char **, char **, char **, char **, char **,
                     int *, double *, int *, char **);

void Rprintf(const char *format, ...) {
    va_list args; va_start(args, format); vfprintf(stderr, format, args); va_end(args);
}
void error(const char *format, ...) {
    va_list args; va_start(args, format); vfprintf(stderr, format, args); va_end(args);
    fputc('\n', stderr); exit(2);
}
void *R_alloc(size_t count, int size) {
    void *result = calloc(count, (size_t)size);
    if (!result) error("Allocation failure");
    return result;
}
void GetRNGstate(void) { }
void PutRNGstate(void) { }
double runif(double a, double b) {
    (void)a; (void)b;
    error("R RNG sampling path unsupported; native C50 requires sampling disabled");
    return 0;
}
static char *read_text(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) error("Cannot open input file");
    if (fseek(file, 0, SEEK_END)) error("Cannot seek input");
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET)) error("Cannot read input size");
    char *text = R_alloc((size_t)size + 1, 1);
    if (fread(text, 1, (size_t)size, file) != (size_t)size) error("Cannot read input");
    fclose(file); return text;
}
static long integer(const char *text) {
    char *end; errno = 0; long value = strtol(text, &end, 10);
    if (errno || *end || value < 0 || value > INT_MAX / 2) error("Invalid integer argument");
    return value;
}
int main(int argc, char **argv) {
    if (argc != 7) error("Expected fit/predict and five arguments");
    char *names = read_text(argv[2]);
    char *data = read_text(argv[3]);
    char *cost = "", *rules = "", *output = "", *tree = "";
    int trials = 1;
    if (!strcmp(argv[1], "fit")) {
        int subset = 1, rules_flag = 0, utility = 0, winnow = 0;
        int seed = (int)integer(argv[5]), no_global_pruning = 0;
        int min_cases = 2, fuzzy_threshold = 0, early_stopping = 1;
        double sample = 0, cf = strtod(argv[6], NULL);
        if (!isfinite(cf) || cf <= 0 || cf >= 1) error("Invalid confidence factor");
        ugo_c50(&names, &data, &cost, &subset, &rules_flag, &utility,
                &trials, &winnow, &sample, &seed, &no_global_pruning,
                &cf, &min_cases, &fuzzy_threshold, &early_stopping,
                &tree, &rules, &output);
        if (!tree || !*tree) error("C50 did not produce a tree: %s", output);
        FILE *file = fopen(argv[4], "wb");
        if (!file) error("Cannot create tree file");
        if (fwrite(tree, 1, strlen(tree), file) != strlen(tree)) error("Cannot write tree");
        fclose(file);
    } else if (!strcmp(argv[1], "predict")) {
        tree = read_text(argv[4]);
        long n = integer(argv[6]);
        if (!n) error("Empty prediction input");
        long observed = 0;
        for (char *p = data; *p; ++p) if (*p == '\n') ++observed;
        if (observed != n) error("Prediction row count differs from supplied size");
        int *predicted = R_alloc((size_t)n, sizeof(int));
        double *probability = R_alloc((size_t)n * 2, sizeof(double));
        ugo_predictions(&data, &names, &tree, &rules, &cost,
                        predicted, probability, &trials, &output);
        FILE *file = fopen(argv[5], "wb");
        if (!file) error("Cannot create prediction file");
        for (long i = 0; i < n; ++i) {
            double p0 = probability[2*i], p1 = probability[2*i+1];
            if (!isfinite(p0) || !isfinite(p1) || p0 < 0 || p1 < 0 || fabs(p0+p1-1) > 1e-5)
                error("C50 returned invalid probabilities: %s", output);
            fprintf(file, "%.17g,%.17g\n", p0, p1);
        }
        fclose(file);
    } else error("Unknown operation");
    return 0;
}
