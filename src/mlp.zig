const std = @import("std");
const assert = std.debug.assert;

const Vector = struct {
    data: []f32,
    size: usize,

    pub fn init(size: usize, allocator: *std.mem.Allocator) !Vector {
        return Vector{
            .size = size,
            .data = try allocator.alloc(f32, size),
        };
    }

    pub fn deinit(self: Vector, allocator: *std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn randomize(self: Vector) void {
        var rand_impl = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));
        for (self.data) |*row| {
            row.* = rand_impl.random().float(f32) * 10;
        }
    }

    pub fn randomize_seed(self: Vector, seed: u64) void {
        var rand_impl = std.rand.DefaultPrng.init(seed);
        for (self.data) |*row| {
            row.* = rand_impl.random().float(f32) * 10;
        }
    }

    pub fn dot_Matrix_Vector(self: Vector, left: Vector, right: Matrix) void {
        // Check if outer dimensions of the left and right matrix are the same.
        assert(left.size == right.rows);

        // assert(self.cols == left.rows);
        // assert(self.rows == right.size);

    }

    pub fn print(self: Vector) void {
        std.debug.print("Vector size: {}\n", .{self.size});
        for (self.data) |row| {
            std.debug.print("{d:.1} ", .{row});
            std.debug.print("\n", .{});
        }
    }
};

const Matrix = struct {
    data: [][]f32,
    rows: usize,
    cols: usize,

    pub fn init(columns: usize, rows: usize, allocator: *std.mem.Allocator) !Matrix {
        var data_temp: [][]f32 = undefined;
        data_temp = try allocator.alloc([]f32, rows);
        for (data_temp) |*row| {
            row.* = try allocator.alloc(f32, columns);
        }

        return Matrix{
            .rows = rows,
            .cols = columns,
            .data = data_temp,
        };
    }

    pub fn deinit(self: Matrix, allocator: *std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn randomize(self: Matrix) void {
        var rand_impl = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));
        for (self.data) |row| {
            for (row) |*col| {
                col.* = rand_impl.random().float(f32) * 10;
            }
        }
    }

    pub fn randomize_seed(self: Matrix, seed: u64) void {
        var rand_impl = std.rand.DefaultPrng.init(seed);
        for (self.data) |row| {
            for (row) |*col| {
                col.* = rand_impl.random().float(f32) * 10;
            }
        }
    }

    pub fn dot_Matrix_Matrix(self: Matrix, left: Matrix, right: Matrix) void {
        // Check if outer dimensions of the left and right matrix are the same.
        assert(left.cols == right.rows);

        assert(self.cols == left.rows);
        assert(self.rows == right.cols);
    }

    pub fn dot_Matrix_Vector(self: Matrix, left: Vector, right: Matrix) void {
        // Check if outer dimensions of the left and right matrix are the same.
        // assert(left.rows == right.size);

        // assert(self.cols == left.rows);
        // assert(self.rows == right.size);

    }

    pub fn print(self: Matrix) void {
        std.debug.print("Amount of  Rows: {}, Cols: {}\n", .{ self.rows, self.cols });
        for (self.data) |row| {
            for (row) |col| {
                std.debug.print("{d:.1} ", .{col});
            }
            std.debug.print("\n", .{});
        }
    }
};

pub fn main() !void {
    std.debug.print("Hello world!\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var a = try Matrix.init(10, 5, &allocator);
    // var b = try Matrix.init(5, 10, &allocator);
    var b_vector = try Vector.init(5, &allocator);
    var c = try Matrix.init(5, 5, &allocator);
    // defer m.deinit(&allocator);

    a.print();
    a.randomize_seed(260);
    a.print();

    b_vector.print();
    b_vector.randomize_seed(260);
    b_vector.print();

    c.dot_Matrix_Vector(a, b_vector);
}
