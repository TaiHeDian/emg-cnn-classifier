module CoBCD(CLK,EN,Out1,clk2,Out0);
 input CLK /*synthesis chip_pin="R3" */;
 input clk2 /*synthesis chip_pin="G22" */;
 input EN /*synthesis chip_pin="V4" */;
 output [7:0]Out1 /*synthesis chip_pin="B6,C6,A4,B4,C7,A5,A6,B21"*/;
 output [7:0]Out0 /*synthesis chip_pin="H1,J3,J1,H4,J2,H2,G3,H3"*/;
 reg [3:0]Q1;
 reg [3:0]Q0;
 reg [7:0]O1;
 reg [7:0]O0;
 reg [7:0]Out1;
 reg [7:0]Out0;
 always@(posedge CLK )
  begin    
   if(!EN) begin Q1=0;Q0=0;end
     else 
    begin
     if(Q0==4'b1001)
      begin 
       if(Q1==4'b1001) begin Q1=0;Q0=0;end
       else begin Q1=Q1+1;Q0=0;end 
      end
     else Q0=Q0+1;
    end   
  end
 always@( Q1 or Q0)
  begin  
  case(Q1)
    4'b0000: O1<=8'b00111111;
    4'b0001: O1<=8'b00000110;
    4'b0010: O1<=8'b01011011;
    4'b0011: O1<=8'b01001111;
    4'b0100: O1<=8'b01100110;
    4'b0101: O1<=8'b01101101;
    4'b0110: O1<=8'b01111101;
    4'b0111: O1<=8'b00000111;
    4'b1000: O1<=8'b01111111;
    4'b1001: O1<=8'b01101111;
    default: O1<=8'b00111111;
   endcase
   case(Q0)
    4'b0000: O0<=8'b00111111;
    4'b0001: O0<=8'b00000110;
    4'b0010: O0<=8'b01011011;
    4'b0011: O0<=8'b01001111;
    4'b0100: O0<=8'b01100110;
    4'b0101: O0<=8'b01101101;
    4'b0110: O0<=8'b01111101;
    4'b0111: O0<=8'b00000111;
    4'b1000: O0<=8'b01111111;
    4'b1001: O0<=8'b01101111;
    default: O0<=8'b00111111;
   endcase
  end
 always@(negedge EN )
  begin
     Out1=O0;
     Out0=O1; 
   
  end 
   
endmodule