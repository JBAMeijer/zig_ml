const std = @import("std");

var train_data_xor = [_][3]f32{
    [_]f32{ 0, 0, 0 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 1, 0 },
};

var train_data_and = [_][3]f32{
    [_]f32{ 0, 0, 0 },
    [_]f32{ 1, 0, 0 },
    [_]f32{ 0, 1, 0 },
    [_]f32{ 1, 1, 1 },
};
var train_data_or = [_][3]f32{
    [_]f32{ 0, 0, 0 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 1, 1 },
};

var train_data_nor = [_][3]f32{
    [_]f32{ 0, 0, 1 },
    [_]f32{ 1, 0, 0 },
    [_]f32{ 0, 1, 0 },
    [_]f32{ 1, 1, 0 },
};
var train_data_nand = [_][3]f32{
    [_]f32{ 0, 0, 1 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 1, 0 },
};

const Xor = struct {
    or_w1: f32,
    or_w2: f32,
    or_b: f32,

    nand_w1: f32,
    nand_w2: f32,
    nand_b: f32,

    and_w1: f32,
    and_w2: f32,
    and_b: f32,

    train_data: [][3]f32,

    pub fn init(seed_number: u64, train_data: [][3]f32) Xor {
        var prng = std.Random.DefaultPrng.init(seed_number);
        const rand = prng.random();
        return Xor{
            .or_w1 = rand.float(f32),
            .or_w2 = rand.float(f32),
            .or_b = rand.float(f32),
            .nand_w1 = rand.float(f32),
            .nand_w2 = rand.float(f32),
            .nand_b = rand.float(f32),
            .and_w1 = rand.float(f32),
            .and_w2 = rand.float(f32),
            .and_b = rand.float(f32),
            .train_data = train_data,
        };
    }

    pub fn print(self: Xor) void {
        std.debug.print("or_w1 = {d:.6}\n", .{self.or_w1});
        std.debug.print("or_w2 = {d:.6}\n", .{self.or_w2});
        std.debug.print("or_b = {d:.6}\n", .{self.or_b});
        std.debug.print("nand_w1 = {d:.6}\n", .{self.nand_w1});
        std.debug.print("nand_w2 = {d:.6}\n", .{self.nand_w2});
        std.debug.print("nand_b = {d:.6}\n", .{self.nand_b});
        std.debug.print("and_w1 = {d:.6}\n", .{self.and_w1});
        std.debug.print("and_w2 = {d:.6}\n", .{self.and_w2});
        std.debug.print("and_b = {d:.6}\n", .{self.and_b});
    }

    pub fn cost(self: Xor) f32 {
        var result: f32 = 0.0;
        for (self.train_data) |slice| {
            const x1 = slice[0];
            const x2 = slice[1];
            const y = self.forward(x1, x2);
            const d = y - slice[2];
            result += d * d;
        }
        result /= train_data_xor.len;

        return result;
    }

    pub fn finite_difference(self: Xor, eps: f32) Xor {
        var out_model = self;
        var temp_model = self;

        const c = self.cost();
        var saved: f32 = 0;

        saved = self.or_w1;
        temp_model.or_w1 += eps;
        out_model.or_w1 = (temp_model.cost() - c) / eps;
        temp_model.or_w1 = saved;

        saved = self.or_w2;
        temp_model.or_w2 += eps;
        out_model.or_w2 = (temp_model.cost() - c) / eps;
        temp_model.or_w2 = saved;

        saved = self.or_b;
        temp_model.or_b += eps;
        out_model.or_b = (temp_model.cost() - c) / eps;
        temp_model.or_b = saved;

        saved = self.nand_w1;
        temp_model.nand_w1 += eps;
        out_model.nand_w1 = (temp_model.cost() - c) / eps;
        temp_model.nand_w1 = saved;

        saved = self.nand_w2;
        temp_model.nand_w2 += eps;
        out_model.nand_w2 = (temp_model.cost() - c) / eps;
        temp_model.nand_w2 = saved;

        saved = self.nand_b;
        temp_model.nand_b += eps;
        out_model.nand_b = (temp_model.cost() - c) / eps;
        temp_model.nand_b = saved;

        saved = self.and_w1;
        temp_model.and_w1 += eps;
        out_model.and_w1 = (temp_model.cost() - c) / eps;
        temp_model.and_w1 = saved;

        saved = self.and_w2;
        temp_model.and_w2 += eps;
        out_model.and_w2 = (temp_model.cost() - c) / eps;
        temp_model.and_w2 = saved;

        saved = self.and_b;
        temp_model.and_b += eps;
        out_model.and_b = (temp_model.cost() - c) / eps;
        temp_model.and_b = saved;

        return out_model;
    }

    pub fn learn(self: Xor, g: Xor, rate: f32) Xor {
        var out = self;
        out.or_w1 -= rate * g.or_w1;
        out.or_w2 -= rate * g.or_w2;
        out.or_b -= rate * g.or_b;
        out.nand_w1 -= rate * g.nand_w1;
        out.nand_w2 -= rate * g.nand_w2;
        out.nand_b -= rate * g.nand_b;
        out.and_w1 -= rate * g.and_w1;
        out.and_w2 -= rate * g.and_w2;
        out.and_b -= rate * g.and_b;
        return out;
    }

    fn forward(self: Xor, x1: f32, x2: f32) f32 {
        // first layer
        const a = sigmoidf(self.or_w1 * x1 + self.or_w2 * x2 + self.or_b);
        const b = sigmoidf(self.nand_w1 * x1 + self.nand_w2 * x2 + self.nand_b);
        // second layer
        return sigmoidf(self.and_w1 * a + self.and_w2 * b + self.and_b);
    }
};

pub fn main() void {
    const eps = 1e-1;
    const rate = 1e-1;

    var model = Xor.init(2678, train_data_xor[0..]);
    model.print();

    for (0..1000000) |_| {
        const b = model.finite_difference(eps);
        model = model.learn(b, rate);
        // std.debug.print("cost = {d:.6}\n", .{model.cost()});
    }

    std.debug.print("-----------------------------\n", .{});
    std.debug.print("cost = {d:.6}\n", .{model.cost()});
    model.print();

    std.debug.print("-----------------------------\n", .{});
    {
        var i: usize = 0;
        while (i < 2) : (i += 1) {
            var j: usize = 0;
            while (j < 2) : (j += 1) {
                std.debug.print("{} - {} = {d:.6}\n", .{ i, j, model.forward(@floatFromInt(i), @floatFromInt(j)) });
            }
        }
    }
    std.debug.print("-----------------------------\n", .{});
    {
        var i: usize = 0;
        while (i < 2) : (i += 1) {
            var j: usize = 0;
            while (j < 2) : (j += 1) {
                std.debug.print("{} - {} = {d:.6}\n", .{ i, j, sigmoidf(model.or_w1 * @as(f32, @floatFromInt(i)) + model.or_w2 * @as(f32, @floatFromInt(j)) + model.or_b) });
            }
        }
    }
    std.debug.print("-----------------------------\n", .{});
    {
        var i: usize = 0;
        while (i < 2) : (i += 1) {
            var j: usize = 0;
            while (j < 2) : (j += 1) {
                std.debug.print("{} - {} = {d:.6}\n", .{ i, j, sigmoidf(model.nand_w1 * @as(f32, @floatFromInt(i)) + model.nand_w2 * @as(f32, @floatFromInt(j)) + model.nand_b) });
            }
        }
    }
    std.debug.print("-----------------------------\n", .{});
    {
        var i: usize = 0;
        while (i < 2) : (i += 1) {
            var j: usize = 0;
            while (j < 2) : (j += 1) {
                std.debug.print("{} - {} = {d:.6}\n", .{ i, j, sigmoidf(model.and_w1 * @as(f32, @floatFromInt(i)) + model.and_w2 * @as(f32, @floatFromInt(j)) + model.and_b) });
            }
        }
    }
}

fn sigmoidf(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}
