const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const twice_exe = b.addExecutable(.{
        .name = "zig_ml_twice",
        .root_source_file = b.path("src/twice.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(twice_exe);

    const gates_exe = b.addExecutable(.{
        .name = "zig_ml_gates",
        .root_source_file = b.path("src/gates.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(gates_exe);

    const gate_xor_exe = b.addExecutable(.{
        .name = "zig_ml_gates_xor",
        .root_source_file = b.path("src/gates_xor.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(gate_xor_exe);

    const run_cmd = b.addRunArtifact(twice_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("twicerun", "Run the twice app");
    run_step.dependOn(&run_cmd.step);

    const gates_run_cmd = b.addRunArtifact(gates_exe);
    gates_run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        gates_run_cmd.addArgs(args);
    }

    const gates_run_step = b.step("gaterun", "Run the gate app");
    gates_run_step.dependOn(&gates_run_cmd.step);

    const gates_xor_run_cmd = b.addRunArtifact(gate_xor_exe);
    gates_xor_run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        gates_xor_run_cmd.addArgs(args);
    }

    const gates_xor_run_step = b.step("gatexorrun", "Run the gate xor app");
    gates_xor_run_step.dependOn(&gates_xor_run_cmd.step);
}
