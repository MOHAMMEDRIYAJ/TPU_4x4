`timescale 1ns/1ps

module tb_systolic_array;

    // Parameters matching DUT
    parameter N  = 8;
    parameter K  = 9;  // Inner reduction dimension (Matrix A is 8x9, Matrix B is 9x8)
    parameter DW = 8;
    parameter AW = 32;
    parameter CLK_PERIOD = 10;

    // DUT Signals
    reg                         clk;
    reg                         rst;
    reg                         enb;
    reg                         clear_acc;
    reg  signed [N*DW-1:0]      a_in_bus;
    reg  signed [N*DW-1:0]      b_in_bus;
    wire signed [N*N*AW-1:0]    psum_bus;
    integer i;

    // DUT Instantiation
    systolic_array #(
        .N(N),
        .DW(DW),
        .AW(AW)
    ) dut (
        .clk(clk),
        .rst(rst),
        .enb(enb),
        .clear_acc(clear_acc),
        .a_in_bus(a_in_bus),
        .b_in_bus(b_in_bus),
        .psum_bus(psum_bus)
    );

    // Clock Generation
    initial clk = 0;
    always #(CLK_PERIOD / 2.0) clk = ~clk;

    // Storage for the 8x9 and 9x8 input matrices
    reg signed [DW-1:0] mat_A [0:N-1][0:K-1];
    reg signed [DW-1:0] mat_B [0:K-1][0:N-1];

    integer row, col, t;

    initial begin
        // ------------------------------------------------------------
        // 1. Initialize Matrix A (8x9)
        // ------------------------------------------------------------
        mat_A[0][0] = 2; mat_A[0][1] = 5; mat_A[0][2] = 1; mat_A[0][3] = 7; mat_A[0][4] = 3; mat_A[0][5] = 4; mat_A[0][6] = 6; mat_A[0][7] = 0; mat_A[0][8] = 8;
        mat_A[1][0] = 9; mat_A[1][1] = 1; mat_A[1][2] = 4; mat_A[1][3] = 2; mat_A[1][4] = 6; mat_A[1][5] = 3; mat_A[1][6] = 0; mat_A[1][7] = 5; mat_A[1][8] = 7;
        mat_A[2][0] = 3; mat_A[2][1] = 8; mat_A[2][2] = 2; mat_A[2][3] = 1; mat_A[2][4] = 5; mat_A[2][5] = 9; mat_A[2][6] = 4; mat_A[2][7] = 6; mat_A[2][8] = 0;
        mat_A[3][0] = 6; mat_A[3][1] = 2; mat_A[3][2] = 7; mat_A[3][3] = 4; mat_A[3][4] = 1; mat_A[3][5] = 8; mat_A[3][6] = 3; mat_A[3][7] = 5; mat_A[3][8] = 9;
        mat_A[4][0] = 1; mat_A[4][1] = 4; mat_A[4][2] = 9; mat_A[4][3] = 3; mat_A[4][4] = 7; mat_A[4][5] = 2; mat_A[4][6] = 8; mat_A[4][7] = 0; mat_A[4][8] = 5;
        mat_A[5][0] = 5; mat_A[5][1] = 7; mat_A[5][2] = 0; mat_A[5][3] = 6; mat_A[5][4] = 2; mat_A[5][5] = 4; mat_A[5][6] = 9; mat_A[5][7] = 3; mat_A[5][8] = 1;
        mat_A[6][0] = 8; mat_A[6][1] = 3; mat_A[6][2] = 5; mat_A[6][3] = 9; mat_A[6][4] = 0; mat_A[6][5] = 7; mat_A[6][6] = 2; mat_A[6][7] = 4; mat_A[6][8] = 6;
        mat_A[7][0] = 4; mat_A[7][1] = 6; mat_A[7][2] = 1; mat_A[7][3] = 8; mat_A[7][4] = 5; mat_A[7][5] = 0; mat_A[7][6] = 7; mat_A[7][7] = 9; mat_A[7][8] = 2;

        // ------------------------------------------------------------
        // 2. Initialize Matrix B (9x8 Signed)
        // ------------------------------------------------------------
        mat_B[0][0] =  2; mat_B[0][1] = -1; mat_B[0][2] =  3; mat_B[0][3] =  4; mat_B[0][4] = -2; mat_B[0][5] =  5; mat_B[0][6] =  1; mat_B[0][7] = -3;
        mat_B[1][0] = -4; mat_B[1][1] =  2; mat_B[1][2] = -1; mat_B[1][3] =  3; mat_B[1][4] =  5; mat_B[1][5] = -2; mat_B[1][6] =  4; mat_B[1][7] =  1;
        mat_B[2][0] =  1; mat_B[2][1] =  3; mat_B[2][2] = -5; mat_B[2][3] =  2; mat_B[2][4] =  4; mat_B[2][5] =  1; mat_B[2][6] = -3; mat_B[2][7] =  6;
        mat_B[3][0] =  5; mat_B[3][1] = -2; mat_B[3][2] =  4; mat_B[3][3] = -1; mat_B[3][4] =  3; mat_B[3][5] =  6; mat_B[3][6] =  2; mat_B[3][7] = -4;
        mat_B[4][0] = -3; mat_B[4][1] =  1; mat_B[4][2] =  2; mat_B[4][3] =  5; mat_B[4][4] = -4; mat_B[4][5] =  3; mat_B[4][6] =  6; mat_B[4][7] =  2;
        mat_B[5][0] =  4; mat_B[5][1] =  5; mat_B[5][2] = -2; mat_B[5][3] =  1; mat_B[5][4] =  2; mat_B[5][5] = -3; mat_B[5][6] =  7; mat_B[5][7] = -1;
        mat_B[6][0] = -1; mat_B[6][1] =  4; mat_B[6][2] =  6; mat_B[6][3] = -3; mat_B[6][4] =  1; mat_B[6][5] =  2; mat_B[6][6] = -5; mat_B[6][7] =  3;
        mat_B[7][0] =  3; mat_B[7][1] = -4; mat_B[7][2] =  1; mat_B[7][3] =  6; mat_B[7][4] = -2; mat_B[7][5] =  5; mat_B[7][6] =  4; mat_B[7][7] = -1;
        mat_B[8][0] =  2; mat_B[8][1] =  1; mat_B[8][2] = -3; mat_B[8][3] =  4; mat_B[8][4] =  5; mat_B[8][5] = -2; mat_B[8][6] =  1; mat_B[8][7] =  6;

        // ------------------------------------------------------------
        // 3. Reset and Enable
        // ------------------------------------------------------------
        rst       = 0;
        enb       = 0;
        clear_acc = 1;
        a_in_bus  = 0;
        b_in_bus  = 0;

        @(negedge clk);
        rst       = 1;
        @(negedge clk);
        rst       = 0;
        clear_acc = 0;
        enb       = 1;

        // ------------------------------------------------------------
        // 4. Feed Matrices with Diagonal Skew
        // ------------------------------------------------------------
        // Streaming takes (K + N - 1) = 9 + 8 - 1 = 16 clock cycles
        for (t = 0; t < (K + N - 1); t = t + 1) begin
            for (row = 0; row < N; row = row + 1) begin
                // Row i of A is delayed by i clock cycles
                if ((t >= row) && (t - row < K))
                    a_in_bus[row*DW +: DW] = mat_A[row][t - row];
                else
                    a_in_bus[row*DW +: DW] = {DW{1'b0}};
            end

            for (col = 0; col < N; col = col + 1) begin
                // Col j of B is delayed by j clock cycles
                if ((t >= col) && (t - col < K))
                    b_in_bus[col*DW +: DW] = mat_B[t - col][col];
                else
                    b_in_bus[col*DW +: DW] = {DW{1'b0}};
            end

            @(negedge clk);
        end

        // Zero out inputs after streaming completes
        a_in_bus = 0;
        b_in_bus = 0;

        // ------------------------------------------------------------
        // 5. Wait for Array Flush and Accumulation
        // ------------------------------------------------------------
        // Wait for trailing products to reach PE(7,7) and settle
        repeat (N + 2) @(negedge clk);
        enb = 0;
        // ------------------------------------------------------------
        // 6. Print the 8x8 Result Matrix
        // ------------------------------------------------------------
        $display("\n============= RESULT MATRIX (psum_bus) =============");
        for (row = 0; row < N; row = row + 1) begin
            $write("Row %0d: [ ", row);
            for (col = 0; col < N; col = col + 1) begin
                $write("%6d ", $signed(psum_bus[(row*N + col)*AW +: AW]));
            end
            $write("]\n");
        end
        $display("====================================================\n");
        
        @(negedge clk) clear_acc = 1;
        @(negedge clk) clear_acc = 0;
        
        $display("\n============= RESULT MATRIX (psum_bus) =============");
        for (row = 0; row < N; row = row + 1) begin
            $write("Row %0d: [ ", row);
            for (col = 0; col < N; col = col + 1) begin
                $write("%6d ", $signed(psum_bus[(row*N + col)*AW +: AW]));
            end
            $write("]\n");
        end
        $display("====================================================\n");
        
        #20;
        $finish;
    end

endmodule
