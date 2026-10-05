`timescale 1ns/1ps

module pe #(
    parameter DW = 8,
    parameter AW = 32
)(
    input                         clk,
    input                         rst,
    input                         enb,
    input                         clear_acc,

    input  signed [DW-1:0]        a_in,      // Signed 8-bit activation (West)
    input  signed [DW-1:0]        b_in,      // Signed 8-bit weight (North)

    output reg  signed [DW-1:0]   a_out,     // Forwarded signed activation (East)
    output reg  signed [DW-1:0]   b_out,     // Forwarded signed weight (South)
    output reg  signed [AW-1:0]   psum       // Signed 32-bit partial sum
);

    // Combinational signed multiplication: 8-bit signed * 8-bit signed = 16-bit signed
    wire signed [(2*DW)-1:0] mult_comb = a_in * b_in;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            a_out <= {DW{1'b0}};
            b_out <= {DW{1'b0}};
            psum  <= {AW{1'b0}};
        end else begin
            // Systolic boundary operand forwarding registers
            a_out <= a_in;
            b_out <= b_in;

            // Output-Stationary Accumulator
            if (clear_acc) begin
                psum <= {AW{1'b0}};
            end else if (enb) begin
                psum <= psum + mult_comb;
            end
        end
    end

endmodule
