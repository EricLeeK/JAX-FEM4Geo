# -*- coding: utf-8 -*-
#================导入本脚本中使用的重要库===========================#
import numpy as np # 数值Python库，用于实现高效的数值计算
import tkinter, tkinter.filedialog, tkinter.messagebox # GUI模块 "tkinter"
import os  # 用于处理操作系统相关操作，如查找文件路径等
import sys # 用于在发生意外错误时停止模拟


#=====以下项目在本脚本中被视为常量================#
"""
求解二维问题（当前实现仅支持二维问题）
"""
DIM = 2                     
"""
每个三角形单元(TRI3)中的局部节点数量。
仅支持TRI3单元
"""
num_nodes_in_element = 3       
"""
时间步长增量（秒）
请根据CFL条件仔细设置此值。
"""
dt =  1.0e-6                   
"""
您的模拟运行时间范围为 t = 0 ~ dt*number_of_time_steps
"""
number_of_time_steps = 10000  
"""
每隔 "output_timing" 步，生成输出文件（paraview格式）。
请勿使用过小的值
"""
output_timing   = 100    
"""
材料属性的总数量。
如果是均质材料，使用1。如果是非均质材料，根据需要使用(>=1)。
"""
num_matProps    = 1           
"""
杨氏模量 E[Pa]。
需要为 "num_matProps" 数量提供E值。例如：如果 num_matProps ==2，np.array([1.0e9,2.0e9])
"""
YoungE   = np.array([1.0e9])   
"""
泊松比 nyu[--]。
需要为 "num_matProps" 数量提供nyu值。例如：如果 num_matProps ==2，np.array([0.2,0.3])
"""
Poiss    = np.array([0.2])     
"""
固体密度 rho[kg/m^3]。
需要为 "num_matProps" 数量提供rho值。例如：如果 num_matProps ==2，np.array([2500.0,2700.0])
"""
Density  = np.array([2500.0]) 
"""
粘性阻尼系数 eta[Pa sec]。
需要为 "num_matProps" 数量提供eta值。例如：如果 num_matProps ==2，np.array([100.0,10.0]) 
"""
eta_damp = np.array([1000.0])  

"""
当 DoYouAppyTimeDependantPressure 为 True 时，
使用 P(t)= Pressure_max*(1-exp(-Pressure_riseAlpha*t)) 对钻孔施加压力
"""
DoYouAppyTimeDependantPressure = True
Pressure_max       = 100.0e6 #[Pa]
Pressure_riseAlpha = 1.0e5   #>0，值越大表示加载速率越高
# 钻孔压力施加位置
X_cen_borehole     = 0.0     # 钻孔中心坐标
Y_cen_borehole     = 0.0     # 钻孔中心坐标
Borehole_Radius    = 0.025   # 单位：米
buf_detectBorehole   = 1.0e-4  # 用于检测钻孔上节点的缓冲区大小
"""
# 待实现
DoYouAppyAbsorbingBoundary = True
"""

#======= 函数列表（注意：调用这些函数的主程序位于这些函数下方）========#
######################################################################   
######################################################################
######################################################################

def GetInputFileInfo():
    # Tkinter(GUI)（请将以下代码当作一种魔法咒语使用）
    ####（魔法咒语：开始）#####
    root = tkinter.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    fTyp = [("","*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    returunval = False
    while (not returunval):
        tkinter.messagebox.showinfo('读取Gmsh软件生成的2D FEM网格','选择 .msh 文件并按确定')
        filename = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
        message = f'您指定的文件名是: \n{filename}\n???\n 如果是，请按是(Y)'
        returunval = tkinter.messagebox.askyesno('确认！', message)
    ####（魔法咒语：结束）#####
    ####（注意："inputfilename" 已存储您要读取的目标网格文件）#####
    return filename

######################################################################   
######################################################################
######################################################################
def GetTotalNumberOfNodesAndElementsFromGmeshData(filename):
    num_triangles = 0
    num_nodes = 0
    with open(filename, mode = 'r', encoding = 'utf-8') as inpfile:
        # 处理此文件的每一行
        linecount = 0 # 计数器，显示当前正在处理输入文件的哪一行
        for line_in_file in inpfile: # 注意 "line_in_file" 是每行的内容，为字符串数据
            linecount += 1
            if(linecount<=1):   # .msh文件的第1行是废弃数据
                continue
            elif(linecount==2): # .msh文件的第2行是节点数量
                num_nodes = int(line_in_file)                       
            elif(linecount>2 and linecount<=(2+num_nodes)): # 对于每一行存储节点坐标的数据
                continue
            elif(linecount>(2+num_nodes) and linecount<=(2+num_nodes+2)):
                continue
            elif(linecount==(2+num_nodes+2 +1 )):
                num_entities = int(line_in_file)               
            elif(linecount>(2+num_nodes+2 +1) and linecount<=(2+num_nodes+2+1+num_entities)):
                bufferdata = []
                bufferdata = line_in_file.split()
                numcomp = len(bufferdata)
                 
                if(numcomp==8): # msh文件：包含3节点TRI3信息的每一行有8个空格分隔的值
                    ################################
                    num_triangles+=1   
                    ################################
                elif(numcomp==7): # msh文件：包含2节点边信息的每一行有7个空格分隔的值
                    continue
                elif(numcomp==6): # msh文件：包含点实体信息的每一行有6个空格分隔的值
                    continue
                else:
                    print("读取到意外的实体数据-->警告！")  				   
            else:
                continue
            
    ###############################
    if(num_nodes<=0):
        input("num_nodes<=0-->不可能的情况，没有FEM节点-->停止")
        sys.exit()    
    if(num_triangles<=0):
        input("num_triangles<=0-->不可能的情况，没有FEM单元-->停止")
        sys.exit()    
    ################################    
    return num_nodes, num_triangles
    ################################

######################################################################   
######################################################################
######################################################################
def Read_node_and_connectityFromGmeshData(inputfilename,
                                          num_nodes,
                                          num_triangles,
                                          X_nodes,
                                          x_nodes,
                                          NodeBoundaryFlag,
                                          connectivity,
                                          elprIDs_TRI3):
    
    with open(inputfilename, mode = 'r', encoding = 'utf-8') as inpfile:
        num_nodes_read_so_far = 0
        num_triangles_read_so_far = 0
        # 处理此文件的每一行
        linecount = 0 # 计数器，指示当前正在处理的行
        for line_in_file in inpfile:
            linecount += 1
            if(linecount<=1):   # .msh文件的第1行没有意义
                continue
            elif(linecount==2): # .msh文件的第2行存储节点总数（此处未使用）
                continue
            elif(linecount>2 and linecount<=(2+num_nodes)): # 对于存储节点坐标的每一行
                bufferdata = []
                bufferdata = line_in_file.split()
                numcomp = len(bufferdata)
                if(numcomp==4):                   
                    Xcoord_read_from_file =  float(bufferdata[1])
                    Ycoord_read_from_file =  float(bufferdata[2])
                    # 注意：在t=0时，初始构型和当前构型相同
                    X_nodes[num_nodes_read_so_far,0] = Xcoord_read_from_file
                    X_nodes[num_nodes_read_so_far,1] = Ycoord_read_from_file
                    x_nodes[num_nodes_read_so_far,0] = Xcoord_read_from_file
                    x_nodes[num_nodes_read_so_far,1] = Ycoord_read_from_file
                    ################################
                    num_nodes_read_so_far += 1 
                    ################################
                else:
                    print("输入数据有问题，文件名为",inputfilename ,"\n")
            elif(linecount>(2+num_nodes) and linecount<=(2+num_nodes+2)):
                continue
            elif(linecount==(2+num_nodes+2 +1 )):
                num_entities = int(line_in_file)
            elif(linecount>(2+num_nodes+2 +1) and linecount<=(2+num_nodes+2+1+num_entities)):
                bufferdata = []
                bufferdata = line_in_file.split()
                numcomp = len(bufferdata)
                if(numcomp==8): # msh文件中每个TRI3单元的每一行包含8个空格分隔的信息
                    # 以下行中有-1。
                    # 这是因为在msh文件中全局节点ID从 1, 2, 3, ~~~~~~ 开始，
                    # 而在python中，全局节点ID应该从 0, 1, 2, ~~~~~~ 开始
                    GlobalnodeID0 = int(bufferdata[5])-1
                    GlobalnodeID1 = int(bufferdata[6])-1
                    GlobalnodeID2 = int(bufferdata[7])-1
                    connectivity[num_triangles_read_so_far,0] = GlobalnodeID0
                    connectivity[num_triangles_read_so_far,1] = GlobalnodeID1
                    connectivity[num_triangles_read_so_far,2] = GlobalnodeID2
                    elprIDs_TRI3[num_triangles_read_so_far] = 0 # 您可以为每个单元操作单元属性ID
                    ################################
                    num_triangles_read_so_far += 1   
                    ################################
                elif(numcomp==7): # msh文件中每个两点边实体的每一行包含7个空格分隔的信息
                    # 以下行中有-1。
                    # 这是因为在msh文件中全局节点ID从 1, 2, 3, ~~~~~~ 开始，
                    # 而在python中，全局节点ID应该从 0, 1, 2, ~~~~~~ 开始
                    GlobalnodeID0 = int(bufferdata[5])-1
                    GlobalnodeID1 = int(bufferdata[6])-1
                    NodeBoundaryFlag[GlobalnodeID0] = 1 # GlobalnodeID0 在模型表面上
                    NodeBoundaryFlag[GlobalnodeID1] = 1 # GlobalnodeID1 在模型表面上
                elif(numcomp==6):  # msh文件中每个单点点实体的每一行包含6个空格分隔的信息
                    # print("读取到点实体（本脚本未使用）")
                    continue                    
                else:
                    print("读取到意外的实体数据-->警告！")
  				   
            else:
                continue
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
    if(num_nodes!=num_nodes_read_so_far):
        input("num_nodes!=num_nodes_read_so_far->停止")
        sys.exit()    
    if(num_triangles!=num_triangles_read_so_far):
        input("错误: num_triangles!=num_triangles_read_so_far->停止")
        sys.exit()    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
    return num_nodes, num_triangles
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
    
######################################################################   
######################################################################
######################################################################
def Detect_Edges_on_traction_boundary(num_triangles,
                                      X_nodes,
                                      connectivity,
                                      NodeBoundaryFlag):
    num_edges_on_traction_boudary = 0
    nodeIDs_for_edge_on_traction_boundary= np.empty(0, dtype = np.int64)
    
    for ith_tri in range(num_triangles): # 对于每个三角形单元，从 0 到 num_triangles-1
        for i in range(0,3): # i 从 0,1 到 2    
            j = i+1
            if j > 2: j=0
            nodei = connectivity[ith_tri,i]
            nodej = connectivity[ith_tri,j]
            if(NodeBoundaryFlag[nodei] == 1 and NodeBoundaryFlag[nodej] == 1):   
               
                ###
                Xi = X_nodes[nodei,0]-X_cen_borehole
                Yi = X_nodes[nodei,1]-Y_cen_borehole
                Ri = np.sqrt(Xi*Xi +Yi*Yi)
                ###
                Xj = X_nodes[nodej,0]-X_cen_borehole
                Yj = X_nodes[nodej,1]-Y_cen_borehole
                Rj = np.sqrt(Xj*Xj +Yj*Yj)
                ###
                
                if(np.abs(Ri-Borehole_Radius)<buf_detectBorehole and np.abs(Rj-Borehole_Radius)<buf_detectBorehole): 
                    nodeIDs_for_edge_on_traction_boundary = np.append(nodeIDs_for_edge_on_traction_boundary,(nodei,nodej))
                    num_edges_on_traction_boudary +=1
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$            
    print("*****************************")	
    print(f"已检测到 {num_edges_on_traction_boudary} 条边用于压力牵引边界！！")            
    print("*****************************")	
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$            
    return  num_edges_on_traction_boudary,nodeIDs_for_edge_on_traction_boundary

"""
# 目前未使用
######################################################################   
######################################################################
######################################################################
def Detect_Edges_on_absorbing_boundary(num_triangles,
                                       X_nodes,
                                       connectivity,
                                       NodeBoundaryFlag):
    
    num_edges_on_absorbing_boudary = 0
    nodeIDs_for_edge_on_absorbing_boundary = np.empty(0, dtype = np.int64)
    
    for ith_tri in range(num_triangles): # 对于每个三角形单元，从 0 到 num_triangles-1
        for i in range(0,3): # i 从 0,1 到 2    
            j = i+1
            if j > 2: j=0
            nodei = connectivity[ith_tri,i]
            nodej = connectivity[ith_tri,j]
            if(NodeBoundaryFlag[nodei] == 1 and NodeBoundaryFlag[nodej] == 1):   
               
                ###
                Xi = X_nodes[nodei,0]-X_cen_borehole
                Yi = X_nodes[nodei,1]-Y_cen_borehole
                Ri = np.sqrt(Xi*Xi +Yi*Yi)
                ###
                Xj = X_nodes[nodej,0]-X_cen_borehole
                Yj = X_nodes[nodej,1]-Y_cen_borehole
                Rj = np.sqrt(Xj*Xj +Yj*Yj)
                ###
                
                if(np.abs(Ri)>0.4 and np.abs(Rj)>0.4): 
                    
                    nodeIDs_for_edge_on_absorbing_boundary = np.append(nodeIDs_for_edge_on_absorbing_boundary,(nodei,nodej))
                    num_edges_on_absorbing_boudary +=1
                    print(Ri,Rj)
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$            
    print("*****************************")	
    print(f"已检测到 {num_edges_on_absorbing_boudary} 条边用于吸收边界！！")            
    print("*****************************")	
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$            
    return  num_edges_on_absorbing_boudary,nodeIDs_for_edge_on_absorbing_boundary
"""
######################################################################   
######################################################################
######################################################################
def Set_initial_condition(num_nodes,
                          num_triangles,
                          X_nodes,
                          v_nodes,
                          NodeBoundaryFlag,
                          connectivity,
                          elprIDs_TRI3):
    
    # 您可以在此处操作和设置初始条件（如速度场）
    for ith_node in range(num_nodes):
        X = X_nodes[ith_node,0]
        Y = X_nodes[ith_node,1]
        
######################################################################   
######################################################################
######################################################################
########计算2×2矩阵A的行列式detA和逆矩阵Ainv(2×2)#######
def Compute_det_and_inverseMAT2D(A, Ainv):
    detA = A[0,0]*A[1,1]-A[1,0]*A[0,1]
    Ainv[0,0] = A[1,1]/detA
    Ainv[1,0] =-A[1,0]/detA
    Ainv[0,1] =-A[0,1]/detA
    Ainv[1,1] = A[0,0]/detA
    return detA
######################################################################   
######################################################################
######################################################################
########计算内力{fint}和节点质量#######
def compute_mass_and_fint(num_triangles,
                          connectivity,
                          X_nodes,
                          x_nodes,
                          v_nodes,
                          elprIDs_TRI3,
                          nodal_mass,
                          fint_nodes,
                          sigma_xx,
                          sigma_yy,
                          sigma_xy):
        
    for ith_tri in range(num_triangles): # 对于每个TRI3单元 ID:ith_tri
        matPropID = elprIDs_TRI3[ith_tri] # 此TRI3单元的材料属性ID（如果均质则为0）
        rho = Density[matPropID]      # 此TRI3单元的密度
        E   = YoungE[matPropID]       # 此TRI3单元的杨氏模量
        Nyu = Poiss[matPropID]        # 此TRI3单元的泊松比
        eta    = eta_damp[matPropID]  # 此TRI3单元的阻尼系数eta
        
        # 请根据杨氏模量和泊松比设置拉梅常数
        # https://en.wikipedia.org/wiki/Elastic_modulus
        # 注意：维基百科中的剪切模量G在本代码中是Mu（拉梅常数）
        Lambda = E*Nyu/(1+Nyu)/(1-2*Nyu) # 第一拉梅常数
        Mu     = E/2/(1+Nyu) # 第二拉梅常数（=剪切模量G）
        #######
        F0       = np.zeros((DIM,DIM),dtype=np.float64)
        F0inv    = np.zeros((DIM,DIM),dtype=np.float64)
        FX       = np.zeros((DIM,DIM),dtype=np.float64)
        FXinv    = np.zeros((DIM,DIM),dtype=np.float64)
        LX       = np.zeros((DIM,DIM),dtype=np.float64)
        #######
        nodeIDs  = np.zeros(num_nodes_in_element,dtype=np.int64)
        
        nodeIDs[0:num_nodes_in_element] = connectivity[ith_tri,0:num_nodes_in_element]
       
        if DIM == 2:
            for i in range(1,3,+1):
                F0[0,i-1]=X_nodes[nodeIDs[i],0]-X_nodes[nodeIDs[0],0];  # X0方向
                F0[1,i-1]=X_nodes[nodeIDs[i],1]-X_nodes[nodeIDs[0],1];  # X1方向
                FX[0,i-1]=x_nodes[nodeIDs[i],0]-x_nodes[nodeIDs[0],0];  # X0方向
                FX[1,i-1]=x_nodes[nodeIDs[i],1]-x_nodes[nodeIDs[0],1];  # X1方向
                LX[0,i-1]=v_nodes[nodeIDs[i],0]-v_nodes[nodeIDs[0],0];  # X0方向
                LX[1,i-1]=v_nodes[nodeIDs[i],1]-v_nodes[nodeIDs[0],1];  # X1方向
        elif DIM == 3:
            print("3D未实现")
        
      
        # voli表示初始构型中三角形单元面积的"两倍"
        voli = Compute_det_and_inverseMAT2D(F0,F0inv)
        if voli<=0.0:
            print("警告！！在单元{ith_tri}中发现意外的voli:{voli}")    
            sys.exit()
        
        # 由于voli表示此TRI3面积的"两倍"，rho*voli表示单元质量的两倍。
        # 因此将单元质量的两倍除以6意味着单元质量的1/3。
        # 所以，下一行是将单元质量平均分配给每个TRI3中的三个节点
        nodal_mass[nodeIDs[0:num_nodes_in_element],0] += rho*voli/6.0;
        nodal_mass[nodeIDs[0:num_nodes_in_element],1] += rho*voli/6.0;
        
        # volc表示当前构型中三角形单元面积的"两倍"
        volc = Compute_det_and_inverseMAT2D(FX,FXinv)
        if volc<=0.0:
            print("警告！！在单元{ith_tri}中发现意外的volc:{volc}")    
            sys.exit()
        
        F = FX @ F0inv    # 变形梯度张量 F #
        J = volc/voli     # F张量的行列式（我们也可以这样计算）
        b = F  @ F.T      # 左柯西-格林应变张量 b #
        l = LX @ FXinv    # 速度梯度张量 l #
        d = 0.5*(l+l.T)   # 变形率张量 d #
        # Neo Hookean弹性材料的本构方程（各向同性）
        sigma = Mu/J*(b-np.eye(2,2)) + Lambda*0.5 *(J-1.0/J)*np.eye(2,2) +  2.0*eta*d   # 柯西应力 σ #
       
        # 用于paraview可视化，我们存储柯西应力。0:x方向，1:y方向
        # 主应力在生成输出文件时计算
        sigma_xx[ith_tri] = sigma[0,0]
        sigma_yy[ith_tri] = sigma[1,1]
        sigma_xy[ith_tri] = sigma[0,1] # 或者 =sigma[1,0] 也可以，因为对称性
        
        # 计算并组装内力{fint}
        for i in range(0, 3, +1): # 此循环运行 i=0,1,2
            j=i+1
            if(j>2):j=0
            k=j+1
            if(k>2):k=0
            inode = nodeIDs[i]
            jnode = nodeIDs[j]
            knode = nodeIDs[k]
            ai = x_nodes[knode,1]-x_nodes[jnode,1]
            bi = x_nodes[jnode,0]-x_nodes[knode,0]
           
            fint_nodes[inode,0] += (sigma[0,0]*ai+sigma[0,1]*bi)/2.0
            fint_nodes[inode,1] += (sigma[1,0]*ai+sigma[1,1]*bi)/2.0
         
    # for ith_tri in range(num_triangles) 结束
######################################################################   
######################################################################
######################################################################
def compute_and_apply_time_dependant_pressure_as_fext(ith_timestep,
                                              current_time,
                                              num_edges_on_traction_boudary,
                                              nodeIDs_for_edge_on_traction_boundary,
                                              x_nodes,
                                              fext_nodes):
    
    
    Pressure = Pressure_max * (1-np.exp(-Pressure_riseAlpha*current_time))   
    # 每10个时间步在控制台输出模拟进度
    if ith_timestep%10 ==0:
        print(f"时间步 = {ith_timestep}, t ={current_time*1.0e6:.3f}(微秒)-->压力= {Pressure/1.0e6:.3f} MPa")
   
    for ith_edge_on_traction_boudary in range(num_edges_on_traction_boudary):                        
        node0 = nodeIDs_for_edge_on_traction_boundary[ith_edge_on_traction_boudary,0]
        node1 = nodeIDs_for_edge_on_traction_boundary[ith_edge_on_traction_boudary,1]
        nx = x_nodes[node1,1]-x_nodes[node0,1]
        ny = x_nodes[node0,0]-x_nodes[node1,0]
      
        
        fext_nodes[node0,0] += (-Pressure*nx)/2.0
        fext_nodes[node0,1] += (-Pressure*ny)/2.0
        
        fext_nodes[node1,0] += (-Pressure*nx)/2.0
        fext_nodes[node1,1] += (-Pressure*ny)/2.0    
######################################################################   
######################################################################
######################################################################
"""
# 待实现
def process_absorbing_boundary_condition(num_edges_on_absorbing_boudary,
                                         nodeIDs_for_edge_on_abosorbing_boundary,
                                         x_nodes,
                                         fext_nodes):
    
    for ith_edge_on_traction_boudary in range(num_edges_on_absorbing_boudary):                        
        node0 = nodeIDs_for_edge_on_abosorbing_boundary[ith_edge_on_traction_boudary,0]
        node1 = nodeIDs_for_edge_on_abosorbing_boundary[ith_edge_on_traction_boudary,1]
        nx = x_nodes[node1,1]-x_nodes[node0,1]
        ny = x_nodes[node0,0]-x_nodes[node1,0]
      
        
        fext_nodes[node0,0] += 0
        fext_nodes[node0,1] += 0
        
        fext_nodes[node1,0] += 0
        fext_nodes[node1,1] += 0  
"""        
######################################################################   
######################################################################
######################################################################
# 求解动力学节点运动方程 {M_lump}{a}={fext}-{fint}
# 注意：{fint}前面的"-"号在计算{fint}时已经包含
def Solve_equation_of_motion(num_nodes,
                             nodal_mass,
                             x_nodes,
                             v_nodes,
                             a_nodes,
                             fint_nodes,
                             fext_nodes,
                             NodeBCType):
    
    # 我们只求解不受速度指定边界条件约束的自由节点
    Indice_For_FreeDOF = (NodeBCType[:,:]=='Free')
    a_nodes[Indice_For_FreeDOF] = (fint_nodes[Indice_For_FreeDOF] + fext_nodes[Indice_For_FreeDOF])/nodal_mass[Indice_For_FreeDOF]
    v_nodes[Indice_For_FreeDOF] += a_nodes[Indice_For_FreeDOF] * dt
    x_nodes[Indice_For_FreeDOF] += v_nodes[Indice_For_FreeDOF] * dt           

######################################################################   
######################################################################
######################################################################
# 输出paraview的模拟结果。请将此作为黑盒使用。
# 如果您想了解这里的规则，请直接询问讲师。
def GenerateParaviewOutputFile(ith_timestep,
                               current_time,
                               FileID,
                               num_nodes,
                               num_triangles,
                               x_nodes,
                               v_nodes,
                               NodeBoundaryFlag,
                               connectivity,
                               elprIDs_TRI3,
                               sigma_xx,
                               sigma_yy,
                               sigma_xy):
    
    if(FileID<10):
        outputfilename = "ParaviewOutput00000"+str(FileID)+".vtu"
    elif(FileID<100):
        outputfilename = "ParaviewOutput0000"+str(FileID)+".vtu"
    elif(FileID<1000):
        outputfilename = "ParaviewOutput000"+str(FileID)+".vtu"
    elif(FileID<10000):
        outputfilename = "ParaviewOutput00"+str(FileID)+".vtu"
    elif(FileID<100000):
        outputfilename = "ParaviewOutput0"+str(FileID)+".vtu"
    else:        
        input("你疯了吗？你创建了太多输出文件->停止模拟")
        sys.exit()    
    
    
    with open(outputfilename, mode = 'w', encoding = 'utf-8') as outvtuf:
        # VTU 格式
    	print( "<?xml version=\"1.0\"?> ", file=outvtuf)
    	print( "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\"> ", file=outvtuf)
    	print( "  <UnstructuredGrid> ", file=outvtuf)
    	print( "    <Piece NumberOfPoints= \"",num_nodes,"\" NumberOfCells=\"",num_triangles,"\" > ", file=outvtuf)
    	###############################################/
    	print( "      <PointData Scalars=\"scalars\"> ", file=outvtuf)
    	print( "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">", file=outvtuf)
    	for ith_node in range(num_nodes):
    		print( v_nodes[ith_node,0],v_nodes[ith_node,1],0.0, file=outvtuf)
    	print( "        </DataArray>", file=outvtuf)
    	#----------------------------------------------------------------------------------------------------
    	"""
        print( "        <DataArray type=\"Int32\" Name=\"NodePropertySet(--)\" format=\"ascii\"> ", file=outvtuf)
    	for ith_node in range(num_nodes):
    		print( "      ", NodePropSet[ith_node],file=outvtuf)
    	print( "        </DataArray> ", file=outvtuf)
        """
        #----------------------------------------------------------------------------------------------------
    	print( "        <DataArray type=\"Int32\" Name=\"BoundaryNodeFlag(--)\" format=\"ascii\"> ", file=outvtuf)
    	for ith_node in range(num_nodes):
    		print( "      ", NodeBoundaryFlag[ith_node],file=outvtuf)
    	print( "        </DataArray> ", file=outvtuf)	
    	#----------------------------------------------------------------------------------------------------
    	print( "      </PointData> ", file=outvtuf)
    	
    	print( "      <CellData Scalars=\"TRI3_info\">", file=outvtuf)
        #----------------------------------------------------------------------------------------------------
    	print( "        <DataArray type=\"Int32\" Name=\"ElementPropID\" format=\"ascii\">", file=outvtuf)
    	for ith_tri in range(num_triangles):
    		print(elprIDs_TRI3[ith_tri], file=outvtuf)
    	print( "        </DataArray>", file=outvtuf)
        #----------------------------------------------------------------------------------------------------
    	print( "         <DataArray type=\"Float32\" Name=\"Sigmaxx\" format=\"ascii\">", file=outvtuf)
    	for ith_tri in range(num_triangles):
    		print(sigma_xx[ith_tri], file=outvtuf)
    	print( "        </DataArray>", file=outvtuf)
        #----------------------------------------------------------------------------------------------------
    	print( "         <DataArray type=\"Float32\" Name=\"Sigmayy\" format=\"ascii\">", file=outvtuf)
    	for ith_tri in range(num_triangles):
    		print(sigma_yy[ith_tri], file=outvtuf)
    	print( "         </DataArray>", file=outvtuf)
        #----------------------------------------------------------------------------------------------------
    	print( "         <DataArray type=\"Float32\" Name=\"Sigmaxy\" format=\"ascii\">", file=outvtuf)
    	for ith_tri in range(num_triangles):
    		print(sigma_xy[ith_tri], file=outvtuf)
    	print( "         </DataArray>", file=outvtuf)
        #----------------------------------------------------------------------------------------------------
    	print( "        <DataArray type=\"Float32\" Name=\"sigma1\" format=\"ascii\">", file=outvtuf)
    	for ith_tri in range(num_triangles):
            Temp1 = (sigma_yy[ith_tri] + sigma_xx[ith_tri]) / 2.0 
            Temp2 = np.sqrt((sigma_yy[ith_tri]-sigma_xx[ith_tri]) * (sigma_yy[ith_tri]-sigma_xx[ith_tri]) / 4.0 + sigma_xy[ith_tri] * sigma_xy[ith_tri]); 
            print(Temp1 + Temp2, file=outvtuf)
    	print( "         </DataArray>", file=outvtuf)
        #----------------------------------------------------------------------------------------------------
    	print( "        <DataArray type=\"Float32\" Name=\"sigma3\" format=\"ascii\">", file=outvtuf)
    	for ith_tri in range(num_triangles):
            Temp1 = (sigma_yy[ith_tri] + sigma_xx[ith_tri]) / 2.0 
            Temp2 = np.sqrt((sigma_yy[ith_tri]-sigma_xx[ith_tri]) * (sigma_yy[ith_tri]-sigma_xx[ith_tri]) / 4.0 + sigma_xy[ith_tri] * sigma_xy[ith_tri]); 
            print(Temp1 - Temp2, file=outvtuf)
    	print( "         </DataArray>", file=outvtuf)
        #----------------------------------------------------------------------------------------------------
    	print( "       </CellData>", file=outvtuf)
        ###############################################/
    	print( "      <Points> ", file=outvtuf)
    	#----------------------------------------------------------------------------------------------------
    	print( "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\"> ", file=outvtuf)
    	for ith_node in range(num_nodes):
    		print( "     ", x_nodes[ith_node,0], x_nodes[ith_node,1],0.0, file=outvtuf)
    	print( "        </DataArray> ", file=outvtuf)
    	#----------------------------------------------------------------------------------------------------
    	print( "      </Points> ", file=outvtuf)
    	###############################################/
    	print( "      <Cells> ", file=outvtuf)
    	#----------------------------------------------------------------------------------------------------
    	print( "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\"> ", file=outvtuf)
    	for ith_tri in range(num_triangles):
    		print( "     ", connectivity[ith_tri,0], connectivity[ith_tri,1], connectivity[ith_tri,2], file=outvtuf)
    	print( "        </DataArray> ", file=outvtuf)
    	#----------------------------------------------------------------------------------------------------
    	print( "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\"> ", file=outvtuf)
    	for ith_tri in range(num_triangles):
    		print( "     ", 3*(ith_tri + 1), file=outvtuf)
    	print( "        </DataArray> ", file=outvtuf)
    	#----------------------------------------------------------------------------------------------------
    	print( "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\"> ", file=outvtuf)
    	#----------------------------------------------------------------------------------------------------
    	for ith_tri in range(num_triangles):	
    		print( "                 5 ", file=outvtuf)	# TRI3单元的Cell类型是5
    	print( "        </DataArray> ", file=outvtuf)
    	#----------------------------------------------------------------------------------------------------
    	print( "      </Cells> ", file=outvtuf)
    	###############################################/
    	print( "    </Piece> ", file=outvtuf)
    	print( "  </UnstructuredGrid> ", file=outvtuf)
    	print( "</VTKFile> ", file=outvtuf)    
    
    print("\n***********************************************************************")
    print(f"时间步 {ith_timestep}, 时间 t ={current_time*1.0e6:.3f}(微秒) ->{outputfilename} 已生成！")
    print("***********************************************************************\n")
    
######################################################################   
######################################################################
######################################################################
######################################################################
####################### 主程序 #################################
######################################################################
######################################################################
######################################################################
######################################################################
""" 
祝您在固体动力学显式有限元世界中旅途愉快！这真的很有趣！
    ∩∩                
   （´･ω･）
   ＿| ⊃／(＿＿_
 ／ └-(＿＿＿_／

"""

if DIM != 2:
    input("目前，此代码仅支持二维模拟-->停止")
    sys.exit()        

if num_nodes_in_element != 3:
    input("目前，此代码仅支持3节点三角形单元(TRI3s)-->停止")
    sys.exit()        


####### 使用GUI选择Gmsh文件（网格输入文件）######
inputfilename = GetInputFileInfo()

####### 获取节点总数(=num_nodes)和三角形单元总数(=num_triangles)#######
num_nodes, num_triangles = GetTotalNumberOfNodesAndElementsFromGmeshData(inputfilename)
print("*****************************")					
print("num_nodes（节点数）     = ",num_nodes)
print("num_triangles（三角形数） = ",num_triangles)
print("*****************************")	

#######分配（准备）存储节点信息的数组#######
# 零重置"初始"节点坐标（每个节点有DIM个自由度）
X_nodes    = np.zeros((num_nodes,DIM), dtype = np.float64)       
# 零重置"当前"节点坐标（每个节点有DIM个自由度）
x_nodes    = np.zeros((num_nodes,DIM), dtype = np.float64)       
# 零重置节点速度（每个节点有DIM个自由度）
v_nodes    = np.zeros((num_nodes,DIM), dtype = np.float64)       
# 零重置节点加速度（每个节点有DIM个自由度）
a_nodes    = np.zeros((num_nodes,DIM), dtype = np.float64)      
# 零重置等效于柯西应力张量的内力（x,y方向）
fint_nodes = np.zeros((num_nodes,DIM), dtype = np.float64)       
# 零重置等效于施加压力等的外力（x,y方向）
fext_nodes = np.zeros((num_nodes,DIM), dtype = np.float64)       
# 零重置节点质量
nodal_mass = np.zeros((num_nodes,DIM), dtype = np.float64)  
# 判断节点是在固体内部(0:默认)还是在固体表面边界上(1)的标志
NodeBoundaryFlag = np.zeros(num_nodes, dtype = np.int64) 
# 存储边界条件的标志（"Free（自由，默认）"或"Fixed（固定）"）。
# 当为自由时，节点根据运动方程移动
NodeBCType       = np.full((num_nodes,DIM) ,"Free",dtype=object) 
#######


#######分配（准备）存储TRI3信息的数组#######
# 存储每个TRI3单元中局部三个节点对应的全局节点ID（给-1作为不可能的默认值）
connectivity = (-1)*np.ones((num_triangles,num_nodes_in_element), dtype = np.int64) 
# 每个三角形单元的物理属性ID（0作为默认值）
elprIDs_TRI3 = np.zeros(num_triangles, dtype = np.int64)    
# 每个TRI3单元的柯西应力张量
sigma_xx     = np.zeros(num_triangles, dtype = np.float64)
sigma_yy     = np.zeros(num_triangles, dtype = np.float64)   
sigma_xy     = np.zeros(num_triangles, dtype = np.float64)

####### 从Gmsh文件获取初始节点坐标、模型边界信息、TRI3连接性信息 #######
Read_node_and_connectityFromGmeshData(inputfilename,
                                      num_nodes,
                                      num_triangles,
                                      X_nodes,
                                      x_nodes,
                                      NodeBoundaryFlag,
                                      connectivity,
                                      elprIDs_TRI3)

####### 检测牵引边界上的单元边（请根据需要更改此部分）#######
if DoYouAppyTimeDependantPressure:
    num_edges_on_traction_boudary, nodeIDs_for_edge_on_traction_boundary = Detect_Edges_on_traction_boundary(
                                                                              num_triangles, 
                                                                              X_nodes, 
                                                                              connectivity, 
                                                                              NodeBoundaryFlag)
    # 将 nodeIDs_for_edge_on_traction_boundary 转换为数组 (num_edges_on_traction_boudary,2)
    nodeIDs_for_edge_on_traction_boundary = nodeIDs_for_edge_on_traction_boundary.reshape(-1,2)

"""
if DoYouAppyAbsorbingBoundary:
    num_edges_on_absorbing_boudary, nodeIDs_for_edge_on_absorbing_boundary = Detect_Edges_on_absorbing_boundary(
                                                                              num_triangles, 
                                                                              X_nodes, 
                                                                              connectivity, 
                                                                              NodeBoundaryFlag)
    # 将 nodeIDs_for_edge_on_traction_boundary 转换为数组 (num_edges_on_absorbing_boudary,2)
    nodeIDs_for_edge_on_absorbing_boundary = nodeIDs_for_edge_on_absorbing_boundary.reshape(-1,2)
"""


#######输出初始状态vtu文件供paraview检查:ParaviewOutput000000.vtu #######
FileID = 0
current_time = 0.0
ith_timestep = 0
GenerateParaviewOutputFile(ith_timestep,
                           current_time,
                           FileID,
                           num_nodes,
                           num_triangles,
                           X_nodes,
                           v_nodes,
                           NodeBoundaryFlag,
                           connectivity,
                           elprIDs_TRI3,
                           sigma_xx,
                           sigma_yy,
                           sigma_xy)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 主时间积分时间步从 0 到 number_of_time_steps-1
for ith_timestep in range(0,number_of_time_steps,+1): 
    
    ########计算应力张量、节点力f_int和节点质量#######
    compute_mass_and_fint(num_triangles,
                          connectivity,
                          X_nodes,
                          x_nodes,
                          v_nodes,
                          elprIDs_TRI3,
                          nodal_mass,
                          fint_nodes,
                          sigma_xx,
                          sigma_yy,
                          sigma_xy)
    
    ########计算由时间依赖压力产生的节点力f_ext#######
    if DoYouAppyTimeDependantPressure:   
        compute_and_apply_time_dependant_pressure_as_fext(ith_timestep,
                                                          current_time,
                                                          num_edges_on_traction_boudary,
                                                          nodeIDs_for_edge_on_traction_boundary,
                                                          x_nodes,
                                                          fext_nodes)
    """    
    if DoYouAppyAbsorbingBoundary:
        process_absorbing_boundary_condition(num_edges_on_absorbing_boudary,
                                             nodeIDs_for_edge_on_absorbing_boundary,
                                             x_nodes,
                                             fext_nodes)
        
    """
    
    ########求解运动方程 ma=f_int+f_ext 得到{a}-->更新{v}和{x}#######
    Solve_equation_of_motion(num_nodes,
                             nodal_mass,
                             x_nodes,
                             v_nodes,
                             a_nodes,
                             fint_nodes,
                             fext_nodes,
                             NodeBCType)
    
   
    
    ####### 判断当前时间步是否对应输出时机 ########
    ####### 如果是，生成Paraview可视化输出文件 ########
    if(ith_timestep>0 and ith_timestep % output_timing == 0):
        FileID+=1
        GenerateParaviewOutputFile(ith_timestep,
                                   current_time,
                                   FileID,
                                   num_nodes,
                                   num_triangles,
                                   x_nodes,
                                   v_nodes,
                                   NodeBoundaryFlag,
                                   connectivity,
                                   elprIDs_TRI3,
                                   sigma_xx,
                                   sigma_yy,
                                   sigma_xy)
        
        
    """
    将节点力（fint,fext）和节点质量重置为零，用于下一时间步的计算
    （由于质量守恒定律，节点质量是不变的，所以只需要计算一次）#######
    """
    nodal_mass[0:num_nodes,0:DIM] = 0.0 # 零重置节点质量
    a_nodes[0:num_nodes,0:DIM]    = 0.0 # 零重置节点加速度
    fint_nodes[0:num_nodes,0:DIM] = 0.0 # 零重置每个节点在每个方向的内力{fint}
    fext_nodes[0:num_nodes,0:DIM] = 0.0 # 零重置每个节点在每个方向的外力{fext}
    ########将当前时间推进dt用于下一时间步########
    current_time += dt    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("FEM模拟已完成！您可以通过Paraview查看模拟结果")