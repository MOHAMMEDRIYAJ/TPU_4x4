module pe #(
    parameter DW = 8,      
    parameter AW = 32      
)(
    input                          clk,
    input                          rst,
    input                          clear_acc,   
    input  signed [DW-1:0]         a_in,
    input  signed [DW-1:0]         b_in,
    output reg signed [DW-1:0]     a_out,       
    output reg signed [DW-1:0]     b_out,       
    output reg signed [AW-1:0]     psum
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            a_out <= {DW{1'b0}};
            b_out <= {DW{1'b0}};
            psum  <= {AW{1'b0}};
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            if (clear_acc)
                psum <= {AW{1'b0}};
            else
                psum <= psum + (a_in * b_in);
        end
    end
endmodule

