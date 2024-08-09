const std = @import("std");

// Simpel train data
// first value is input
// second is output
const train_data = [_][2]f32{
    [_]f32{ 0, 0 },
    [_]f32{ 1, 2 },
    [_]f32{ 2, 4 },
    [_]f32{ 3, 6 },
    [_]f32{ 4, 8 },
};

// y = x*w
// x1, x2, x3, ..., b
// w1, w2, w3, ...
// b = bias
// y = x1*w1 + x2*w2 + x3*w3 + ... + b

pub fn main() !void {
    std.debug.print("Hello, World!\n", .{});

    var prng = std.Random.DefaultPrng.init(2678);
    const rand = prng.random();
    var w = rand.float(f32) * 10;
    var b = rand.float(f32) * 5;

    std.debug.print("weight = {d:.6}, bias = {d:.6}\n", .{ w, b });
    const eps = 1e-3;
    const rate = 1e-3;

    std.debug.print("cost start: {d:.3}\n", .{cost(w, b)});
    for (0..1000) |_| {
        const c = cost(w, b);
        const dwcost = (cost(w + eps, b) - c) / eps;
        const dbcost = (cost(w, b + eps) - c) / eps;
        w -= rate * dwcost;
        b -= rate * dbcost;
        std.debug.print("cost = {d:.6}, weight = {d:.6}, bias = {d:.6}\n", .{ cost(w, b), w, b });
    }

    std.debug.print("weight = {d:.6}, bias = {d:.6}\n", .{ w, b });

    const input_data = 12.254;
    std.debug.print("Output: {d:.6}\n", .{input_data * (w + b)});
}

pub fn cost(w: f32, b: f32) f32 {
    var result: f32 = 0.0;
    for (train_data) |slice| {
        const x = slice[0];
        const y = x * w + b;
        const d = y - slice[1];
        result += d * d;
        // std.debug.print("actual output: {d:.3}, expected output: {d:.3}\n", .{ y, slice[1] });
    }
    result /= train_data.len;

    return result;
}

test "simple test" {}
