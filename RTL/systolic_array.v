module systolic_array #(
    parameter N  = 8,
    parameter DW = 8,
    parameter AW = 32
)(
    input                          clk,
    input                          rst,
    input                          enb,
    input                          clear_acc,
    input  signed [N*DW-1:0]       a_in_bus,
    input  signed [N*DW-1:0]       b_in_bus,
    output signed [N*N*AW-1:0]     psum_bus
);
    wire signed [(N*(N+1))*DW-1:0] a_link_flat;
    wire signed [((N+1)*N)*DW-1:0] b_link_flat;

    genvar gi, gj;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : ROW_ENTRY
            assign a_link_flat[(gi*(N+1)+0)*DW +: DW] = a_in_bus[gi*DW +: DW];
        end
        for (gj = 0; gj < N; gj = gj + 1) begin : COL_ENTRY
            assign b_link_flat[(0*N+gj)*DW +: DW] = b_in_bus[gj*DW +: DW];
        end

        for (gi = 0; gi < N; gi = gi + 1) begin : ROWS
            for (gj = 0; gj < N; gj = gj + 1) begin : COLS
                pe #(.DW(DW), .AW(AW)) pe_inst (
                    .clk       (clk),
                    .rst       (rst),
                    .enb       (enb),
                    .clear_acc (clear_acc),
                    .a_in      (a_link_flat[(gi*(N+1)+gj)*DW +: DW]),
                    .b_in      (b_link_flat[(gi*N+gj)*DW +: DW]),
                    .a_out     (a_link_flat[(gi*(N+1)+gj+1)*DW +: DW]),
                    .b_out     (b_link_flat[((gi+1)*N+gj)*DW +: DW]),
                    .psum      (psum_bus[(gi*N+gj)*AW +: AW])
                );
            end
        end
    endgenerate
endmodule
