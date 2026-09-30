module pe #(
    parameter DW = 8,
    parameter AW = 32
)(
    input                       clk,
    input                       rst,
    input                       clear_acc,

    input        [DW-1:0]      a_in,      // Unsigned 8-bit
    input signed [DW-1:0]      b_in,      // Signed 8-bit

    output reg        [DW-1:0] a_out,     // Unsigned 8-bit
    output reg signed [DW-1:0] b_out,     // Signed 8-bit
    output reg signed [AW-1:0] psum       // Signed 32-bit
);

    // Signed 16-bit multiplication result
    reg signed [(2*DW)-1:0] product;

    always @(posedge clk or posedge rst) begin

        if (rst) begin
            a_out <= {DW{1'b0}};
            b_out <= {DW{1'b0}};
            psum  <= {AW{1'b0}};
            product <= {(2*DW){1'b0}};
        end

        else begin

            // Forward operands
            a_out <= a_in;
            b_out <= b_in;

            // Convert unsigned a_in to signed positive value
            // before multiplication.
            product <= $signed({1'b0, a_in}) * b_in;

            // Accumulator
            if (clear_acc)
                psum <= {AW{1'b0}};
            else
                psum <= psum + 
                        ($signed({1'b0, a_in}) * b_in);

        end
    end

endmodule
