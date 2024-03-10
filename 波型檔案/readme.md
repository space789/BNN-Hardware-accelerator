大致流程：
    1.先設定系統參數，如ifmaps大小channel、weight大小channel、function等
    2.以S_AXIS輸入weight
    3.以S_AXIS輸入ifmaps
    4.等待M_AXIS輸出ofmaps

注意事項M_AXIS的輸入皆以5bit對齊M_AXIS_TDATA = {2'd0,data5[4:0],data4[4:0],data3[4:0],data2[4:0],data1[4:0],data0[4:0]}

系統參數設定即流程細節：請配合總波型.png
    AXI為設定參數用，以及CPU輪巡是否完成用。
    接下來先設定kernel size, ofmaps channel input channel, function ofmaps_width(不須照順序)，後輸入write weight start(會在AXI_reg0寫入指令，可參考control unit第1,2行之define)，之後開始輸入weight，輸入完後tb會一直輪巡AXI_reg3是否有32'd1(亦即電路會在寫完weight後會在AXI_reg3寫入1)，檢查到32'd1後給入compute指令開始輸入ifmaps(會在AXI_reg0寫入指令，可參考control unit第1,2行之define)，輸入完ifmaps後會開始輪巡AXI_reg3是否有32'hffffffff(亦即電路會在寫完weight後會在AXI_reg3寫入32'hffffffff)，檢查到後電路即結束。


weight輸入方法：請配合ifmaps weight txt.png,load weight.png
    weight一次輸入一個column多個channel如txt檔所示，因為kernel size只有3所以有兩個bit用不到填0，因此第一筆資料為{2'd0,3'b000}，第二筆為{2'd0,3'b000}，
    第三筆為{2'd0,3'b000}，第四筆為{2'd0,3'b101}，第五筆為{2'd0,3'b010}，第六筆為{2'd0,3'b100}，第七筆為{2'd0,3'b111}。
    而因為AXIS一次最多只可以傳送六筆資料，因此需分兩次。
        第一次{2'd0,{2'd0,3'b100},{2'd0,3'b010},{2'd0,3'b101},{2'd0,3'b000},{2'd0,3'b000},{2'd0,3'b000}} = 32'h08228000
        第二次{27'd0,{2'd0,3'b111}} = 32'h00000007 
        第三次{2'd0,{2'd0,3'b011},{2'd0,3'b101},{2'd0,3'b011},{2'd0,3'b100},{2'd0,3'b011},{2'd0,3'b100}} = 32'h06519064
        第四次{27'd0,{2'd0,3'b110}} = 32'h00000006
        第五次{2'd0,{2'd0,3'b101},{2'd0,3'b111},{2'd0,3'b111},{2'd0,3'b011},{2'd0,3'b010},{2'd0,3'b111}} = 32'h0A738C47
        第六次{27'd0,{2'd0,3'b101}} = 32'h00000005
        完成一個weight的輸入
    故在波型上第一次輸入為32'h08228000第二次為32'h00000007以此類推

ifmaps輸入方法：請配合ifmaps weight txt.png,load ifmaps.png
    和輸入weight同理因此
        第一次{2'd0,{2'd0,3'b011},{2'd0,3'b110},{2'd0,3'b111},{2'd0,3'b000},{2'd0,3'b100},{2'd0,3'b100}} = 32'h06638084
        第二次{27'd0,{2'd0,3'b000}} = 32'h00000000
        第三次{2'd0,{2'd0,3'b010},{2'd0,3'b100},{2'd0,3'b110},{2'd0,3'b000},{2'd0,3'b011},{2'd0,3'b001}} = 32'h04430061
        第四次{27'd0,{2'd0,3'b010}} = 32'h00000002
        第五次{2'd0,{2'd0,3'b110},{2'd0,3'b011},{2'd0,3'b101},{2'd0,3'b101},{2'd0,3'b001},{2'd0,3'b101}} = 32'h0C329425
        第六次{27'd0,{2'd0,3'b110}} = 32'h00000006
        完成一次ifmaps的輸入
    故在波型上第一次輸入為32'h06638084第二次為32'h00000000以此類推

核心計算方法：請配合M_AXIS_TDATA.png, ofmaps txt.png
    load一組ifmaps後和所有的weight做convolution(ifmaps reuse)，因此ofmaps的輸入會是以不同channel同個位置先輸出。
    以3個3kernel為例M_AXIS_TDATA{....ofmaps[0][0][2],ofmaps[2][0][1],ofmaps[1][0][1],ofmaps[0][0][1],ofmaps[2][0][0],ofmaps[1][0][0],ofmaps[0][0][0]}
    因此第一次的輸出是32'b11_111_000_101_011_011_001_100_111_010_101 = 32'hf8ad99d5
    故在波型上第一次輸出為32'hf8ad99d5以此類推
