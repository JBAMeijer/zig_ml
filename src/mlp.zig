const std = @import("std");

const Matrix = struct {
    data: []f32,
    rows: usize,
    cols: usize,

    pub fn init(columns: usize, rows: usize, allocator: *std.mem.Allocator) !Matrix {
        return Matrix{
            .rows = rows,
            .cols = columns,
            .data = try allocator.alloc(f32, rows * columns),
        };
    }

    pub fn deinit(self: Matrix, allocator: *std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn print(self: Matrix) void {
        std.debug.print("Amount of Cols: {}, Rows: {}\n", .{ self.cols, self.rows });
        for (self.data, 1..) |value, index| {
            std.debug.print("{d:.1} ", .{value});
            if (index % self.cols == 0) {
                std.debug.print("\n", .{});
            }
        }
    }
};

pub fn main() !void {
    std.debug.print("Hello world!\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var m = try Matrix.init(9, 32, &allocator);
    defer m.deinit(&allocator);

    m.print();
}
