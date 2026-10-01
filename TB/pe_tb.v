`timescale 1ns/1ps

module pe_tb;

    parameter DW = 8;
    parameter AW = 32;

    // DUT inputs
    reg                   clk;
    reg                   rst;
    reg                   enb;
    reg                   clear_acc;
    reg signed [DW-1:0]   a_in;
    reg signed [DW-1:0]   b_in;

    // DUT outputs
    wire signed [DW-1:0]  a_out;
    wire signed [DW-1:0]  b_out;
    wire signed [AW-1:0]  psum;

    // Instantiate DUT
    pe #(
        .DW(DW),
        .AW(AW)
    ) dut (
        .clk       (clk),
        .rst       (rst),
        .enb       (enb),
        .clear_acc (clear_acc),
        .a_in      (a_in),
        .b_in      (b_in),
        .a_out     (a_out),
        .b_out     (b_out),
        .psum      (psum)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test stimulus
    initial begin

        // Initial values
        rst       = 1;
        clear_acc = 0;
        a_in      = 0;
        b_in      = 0;
        enb       = 0;
        // Apply reset
        #12;
        rst = 0;
        
        @(negedge clk);
        enb = 1;
        a_in = 8'd5;
        b_in = -8'sd2;
        #20;
        @(negedge clk)
        clear_acc = 1;
        @(negedge clk);
        clear_acc = 0;
        a_in = 8'd255;
        b_in = -8'sd1;
        #20;
        @(negedge clk)
        clear_acc = 1;
        @(negedge clk);
        clear_acc = 0;
        a_in = 8'd128;
        b_in = -8'sd128;
        @(negedge clk);
        a_in = 8'd84;
        b_in = 8'sd1;
        @(negedge clk);
        enb = 0;
        #10;
        $finish;
    end

endmodule
