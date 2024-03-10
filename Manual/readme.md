大致流程：
    1.先設定系統參數，如ifmaps大小channel、weight大小channel、function等
    2.以S_AXIS輸入weight
    3.以S_AXIS輸入bias（if any）
    4.以S_AXIS輸入ifmaps
    5.等待M_AXIS輸出ofmaps

系統參數設定即流程細節：請配合AXI_AXIS_timing.png、AXI時序圖.jpg
    AXI為設定參數用，以及CPU輪巡是否完成用。
    接下來先設定kernel size, ofmaps channel input channel, function ofmaps_width(不須照順序)，後輸入write weight start(會在AXI_reg0寫入指令，可參考control unit開頭之define)，之後開始輸入weight，輸入完後tb會一直輪巡AXI_reg3是否有32'd1(亦即電路會在寫完weight後會在AXI_reg3寫入1)，後輸入write bias start(會在AXI_reg0寫入指令，可參考control unit開頭之define)，之後開始輸入bias，輸入完後tb會一直輪巡AXI_reg3是否有32'd2(亦即電路會在寫完weight後會在AXI_reg3寫入2)，檢查到32'd2後給入compute指令開始輸入ifmaps(會在AXI_reg0寫入指令，可參考control unit第1,2行之define)，輸入完ifmaps後會開始輪巡AXI_reg3是否有32'hffffffff(亦即電路會在寫完weight後會在AXI_reg3寫入32'hffffffff)，檢查到後電路即結束。


weight輸入方法：
    weight一次輸入最多32個channel，以先輸入整個column為順序，因此第一筆資料為{29'd0,3'b101}，第二筆為{29'd0,3'b010}，
    第三筆為{29'd0,3'b011}，第四筆為{29'd0,3'b111}，第五筆為{29'd0,3'b100}，第六筆為{29'd0,3'b111}，第七筆為{29'd0,3'b001}，第八筆為{29'd0,3'b100}，第九筆為{29'd0,3'b101}。

bias輸入方法：請配合bias.png
    bias一次輸入一個bias如txt檔所示，且有正負號共16bit（在儲存成npy檔案時為np.int16須注意轉檔），因此第一筆資料為{16'd0,16'hfffd}，第二筆為{16'd0,16'h0000}，
    第三筆為{16'd0,16'hfffb}，以此類推

ifmaps輸入方法：請配合ifmaps weight txt.png,load ifmaps.png
    和輸入weight同理因此
        第一次{29'd0,3'b000} = 32'h0
        第一次{29'd0,3'b010} = 32'h2
        第一次{29'd0,3'b100} = 32'h4
        第一次{29'd0,3'b001} = 32'h1
        第一次{29'd0,3'b111} = 32'h7
        第一次{29'd0,3'b010} = 32'h2
        第一次{29'd0,3'b110} = 32'h6
        第一次{29'd0,3'b010} = 32'h2
        第一次{29'd0,3'b000} = 32'h0
        

核心計算方法：請配合AXI_AXIS_timing.png、ALU_control_signal.jpg
    load一組ifmaps後和所有的weight做convolution(ifmaps reuse)，因此ofmaps的輸入會是以不同channel同個位置先輸出。
    以3個3kernel為例M_AXIS_TDATA{....ofmaps[6][0][0],ofmaps[5][0][0],ofmaps[4][0][0],ofmaps[3][0][0],ofmaps[2][0][0],ofmaps[1][0][0],ofmaps[0][0][0]}
    因此第一次的輸出是32'b11110000110100010111001010001110 = 32'hf0d1728e
    故在波型上第一次輸出為32'hf0d1728e以此類推。
    M_AXIS不會跨ifmaps輸出假設ofmaps_channel = 3 則每次AXIS的有效位數僅有3bit不會將輸出累積至32bit才輸出，pooling亦同。
