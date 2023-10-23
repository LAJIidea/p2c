// #include <stdio.h>

// // 创建一个结构体，其中包含函数指针和一个变量
// struct Closure {
//     int x;
//     int (*add)(int);
// };

// // 创建一个函数，返回一个 Closure 结构体
// struct Closure makeClosure(int x) {
//     // 定义一个局部变量
//     int y = 10;
//     // 定义一个函数指针，指向一个 lambda 表达式
//     int (*add)(int) = [](int z) { return x + y + z; };
//     // 将变量和函数指针打包成一个 Closure 结构体并返回
//     return {x, add};
// }

// int main() {
//     // 创建一个闭包
//     struct Closure c = makeClosure(20);
//     // 调用闭包中的函数
//     printf("%d\n", c.add(30)); // 输出 60
//     return 0;
// }

#include <stdio.h>
#include <stdlib.h>

typedef struct closure {
    void (*fun)(struct closure* cl);
    int data;
} closure;

void closure_func(closure *cl) {
    printf("%d\n", cl->data);
}

closure* make_closure(void (*fun)(closure*), int data) {
    closure* new_closure = malloc(sizeof(closure));
    new_closure->fun = fun;
    new_closure->data = data;
    return new_closure;
}

int main() {
    closure* test = make_closure(closure_func, 42);
    test->fun(test);
    return 0;
}