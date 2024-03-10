`timescale  1ns / 1ps    

module tb_axis_master;   

// axis_master Parameters
parameter PERIOD                = 10;
parameter FIFO_DEPTH            = 4 ;
parameter C_M_AXIS_TDATA_WIDTH  = 32;

// axis_master Inputs
reg   M_AXIS_ACLK                          = 0 ;
reg   M_AXIS_ARESETN                       = 0 ;
reg   [C_M_AXIS_TDATA_WIDTH-1:0]  TDATA_in = 0 ;
reg   TVALID_in                            = 0 ;
reg   TLAST_in                             = 0 ;
reg   M_AXIS_TREADY                        = 0 ;

// axis_master Outputs
wire  [C_M_AXIS_TDATA_WIDTH-1:0]  M_AXIS_TDATA ;
wire  M_AXIS_TVALID                        ;
wire  M_AXIS_TLAST                         ;
wire  [(C_M_AXIS_TDATA_WIDTH/8)-1 : 0]  M_AXIS_TSTRB ;


initial
begin
    forever #(PERIOD/2)  M_AXIS_ACLK=~M_AXIS_ACLK;
end

axis_master #(
    .FIFO_DEPTH           ( FIFO_DEPTH           ),
    .C_M_AXIS_TDATA_WIDTH ( C_M_AXIS_TDATA_WIDTH ))
 u_axis_master (
    .M_AXIS_ACLK             ( M_AXIS_ACLK                                      ),
    .M_AXIS_ARESETN          ( M_AXIS_ARESETN                                   ),
    .TDATA_in                ( TDATA_in        [C_M_AXIS_TDATA_WIDTH-1:0]       ),
    .TVALID_in               ( TVALID_in                                        ),
    .TLAST_in                ( TLAST_in                                         ),
    .M_AXIS_TREADY           ( M_AXIS_TREADY                                    ),

    .M_AXIS_TDATA            ( M_AXIS_TDATA    [C_M_AXIS_TDATA_WIDTH-1:0]       ),
    .M_AXIS_TVALID           ( M_AXIS_TVALID                                    ),
    .M_AXIS_TLAST            ( M_AXIS_TLAST                                     ),
    .M_AXIS_TSTRB            ( M_AXIS_TSTRB    [(C_M_AXIS_TDATA_WIDTH/8)-1 : 0] )
);

always @(*) begin
    if(u_axis_master.M_AXIS_TVALID && M_AXIS_TREADY)
        $display($time,"  M_AXIS_TDATA = %d , M_AXIS_TVALID = %d , M_AXIS_TLAST = %h",u_axis_master.M_AXIS_TDATA,u_axis_master.M_AXIS_TVALID,u_axis_master.M_AXIS_TLAST);
end

always @(negedge M_AXIS_ACLK ) begin
    // M_AXIS_TREADY <= ~M_AXIS_TVALID;
    M_AXIS_TREADY <= 1'd1;
end

integer i;
initial
begin
    M_AXIS_ACLK = 0;
    M_AXIS_ARESETN = 0;
    #(PERIOD*2) M_AXIS_ARESETN  =  1;
    #(PERIOD*2)
    for(i=0;i<32;i=i+1) begin
        if(i==31) begin
            AXIS_input(i,1,1);
        end
        else begin
            AXIS_input(i,0,1);
        end
        #(PERIOD*1);
    end
    TVALID_in = 1'd0;
    #1000;
    $finish;
end

task AXIS_input(input [31:0] data, input last,input continue);begin
        TDATA_in = data;
        TLAST_in = last;
        if(continue) begin
            TVALID_in = 1'd1;
        end
        else begin
            TVALID_in = 1'd1;
            #(PERIOD)
            TVALID_in = 1'd0;
        end
    end
endtask

endmodule