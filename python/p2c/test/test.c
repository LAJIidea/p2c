#include <stdio.h>
#include <stdlib.h>

// Define a struct to hold the captured variables
typedef struct {
    int c;
} ClosureContext;

// Define the inner function
float inner(float m, float n, ClosureContext* ctx) {
    return m * n + ctx->c;
}

// Define the outer function
float (*outer(int a, int b))(float, float, ClosureContext*) {
    // Capture the variable 'c'
    ClosureContext* ctx = (ClosureContext*)malloc(sizeof(ClosureContext));
    if (ctx == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }
    ctx->c = a * b + b;

    // Return a pointer to the inner function along with the captured context
    return inner;
}

int main() {
    int a = 3;
    int b = 4;

    // Create a closure by calling the outer function
    float (*closure)(float, float, ClosureContext*) = outer(a, b);

    // Call the closure
    ClosureContext ctx;
    float result = closure(2.5, 3.5, &ctx);

    printf("Result: %f\n", result);

    // Free the captured context
    free(&ctx);

    return 0;
}
