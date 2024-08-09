const std = @import("std");

// OR-Gate
const train_data_or = [_][3]f32{
    [_]f32{ 0, 0, 0 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 1, 1 },
};

// AND-Gate
const train_data_and = [_][3]f32{
    [_]f32{ 0, 0, 0 },
    [_]f32{ 1, 0, 0 },
    [_]f32{ 0, 1, 0 },
    [_]f32{ 1, 1, 1 },
};

// NAND-Gate
const train_data_nand = [_][3]f32{
    [_]f32{ 0, 0, 1 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 1, 0 },
};

const GateType = enum {
    AND,
    OR,
    NAND,
};

pub fn main() !void {
    train_or_and_nand(.NAND);
}

pub fn train_or_and_nand(gate: GateType) void {
    // const stdocut = std.io.getStdOut().writer();
    // std.debug.print("Hello world\n", .{});

    // var i: f32 = -100.0;
    // while (i <= 100.0) : (i += 1.0) {
    //     std.debug.print("{d:.2} => {d:.20}\n", .{ i, sigmoidf(i) });
    // }
    const train_data = switch (gate) {
        GateType.AND => &train_data_and,
        GateType.OR => &train_data_or,
        GateType.NAND => &train_data_nand,
    };

    var prng = std.Random.DefaultPrng.init(2678);
    const rand = prng.random();
    var w1 = rand.float(f32);
    var w2 = rand.float(f32);
    var b = rand.float(f32);

    const eps = 1e-1;
    const rate = 1e-1;

    for (0..2000000) |_| {
        // try stdout.print("{d:.6}\n", .{cost(w1, w2, b)});
        const c = cost(train_data, w1, w2, b);
        const dw1 = (cost(train_data, w1 + eps, w2, b) - c) / eps;
        const dw2 = (cost(train_data, w1, w2 + eps, b) - c) / eps;
        const db = (cost(train_data, w1, w2, b + eps) - c) / eps;
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
    }

    std.debug.print("w1 = {d:.6}, w2 = {d:.6}, b = {d:.6}, c = {d:.6}\n", .{ w1, w2, b, cost(train_data, w1, w2, b) });
    // try stdout.print("{d:.6}\n", .{cost(w1, w2, b)});
    {
        var i: usize = 0;

        while (i < 2) : (i += 1) {
            var j: usize = 0;
            while (j < 2) : (j += 1) {
                switch (gate) {
                    GateType.OR => std.debug.print("{} | {} = {d:.6}\n", .{ i, j, sigmoidf(@as(f32, @floatFromInt(i)) * w1 + @as(f32, @floatFromInt(j)) * w2 + b) }),
                    GateType.AND => std.debug.print("{} & {} = {d:.6}\n", .{ i, j, sigmoidf(@as(f32, @floatFromInt(i)) * w1 + @as(f32, @floatFromInt(j)) * w2 + b) }),
                    GateType.NAND => std.debug.print("{} ~& {} = {d:.6}\n", .{ i, j, sigmoidf(@as(f32, @floatFromInt(i)) * w1 + @as(f32, @floatFromInt(j)) * w2 + b) }),
                }
            }
        }
    }
}

pub fn cost(train_data: anytype, w1: f32, w2: f32, b: f32) f32 {
    var result: f32 = 0.0;
    for (train_data) |slice| {
        const x1 = slice[0];
        const x2 = slice[1];
        const y = sigmoidf(x1 * w1 + x2 * w2 + b);
        const d = y - slice[2];
        result += d * d;
    }
    result /= train_data.len;

    return result;
}

pub fn sigmoidf(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}
