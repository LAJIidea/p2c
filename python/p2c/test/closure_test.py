from typing import Any

def outer(a: int, b: int) -> Any:
    c = a * b + b
    def inner(m: float, n: float):
        return m * n + c
    return inner

# #include <stdio.h>
# #include <stdlib.h>

# typedef struct closure_s{
#     int x;
#     void (*call)(struct closure_s*);
# } closure;

# void f(closure *clo) {
#     clo->x += 1;
#     printf("node = %d\n", clo->x);
# }

# closure *extent() {
#     closure *func = (closure *)malloc(sizeof(closure));
#     func->x = 0;
#     func->call = f; 
#     return func;
# }

# int main() {
#     closure *clo = extent();
#     clo->call(clo);			// node = 1
#     clo->call(clo);			// node = 2
#     free(clo);
# }
