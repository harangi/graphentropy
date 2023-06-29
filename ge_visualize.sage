lbl_list=[chr(i) for i in range(ord('a'),ord('z')+1)]+[chr(i) for i in range(ord('A'),ord('Z')+1)]+[chr(i) for i in range(ord('0'),ord('9')+1)]+[chr(i) for i in range(192,500)]
lbl_map={i:lbl_list[i] for i in range(len(lbl_list))}

def grid(nr_x,nr_y):
    gr=Graphics()
    for x in range(nr_x+1):
        gr+=line([(x-0.5,1),(x-0.5,0.5-nr_y)],thickness=1,color='black')
    for y in range(nr_y+1):
        gr+=line([(-0.5,0.5-y),(nr_x,0.5-y)],thickness=1,color='black')
    return gr    
    
def ge_visualize(ge,show_bool=True,grid_bool=True,bars_bool=True,entr_bool=True,use_amax=True):
    a=ge.a
    amax=np.max(a) if use_amax else 1.
    r=ge.r.sum(axis=1) if ge.cond else ge.r
    sets=ge.sets
    nr_x=ge.nr_x
    nr_j=ge.nr_j
    lbl=ge.lbl
    sets_lbl=[''.join([lbl[i] for i in range(len(s)) if s[i]==1]) for s in sets]

    h0=2.
    th=5.
    crad=0.3

    def a_bar(x,aval):
        return polygon([(x-crad,h0),(x+crad,h0),(x+crad,h0+th*aval/amax),(x-crad,h0+th*aval/amax)],color='red')
        #+text("{:.4f}".format(aval),(x,h0+th+1), rotation="vertical",vertical_alignment='bottom',color='red')

    def a_level(rel_h):
        fig=Graphics()
        fig+=line([(-0.5,h0+th*rel_h),(nr_x-0.5,h0+th*rel_h)],thickness=1,color='grey')
        fig+=text("{:.4f}".format(amax*rel_h),(nr_x+0.5,h0+th*rel_h),horizontal_alignment='left',color='grey')
        return fig

    def ci(x,y):
        return circle((x,-y), crad, fill=True, color='blue')
        
    fig=Graphics()    
    
    for x in range(nr_x):
        fig+=text(lbl[x],(x,1.2),color='black')
        
    if bars_bool:
        for rel_h in [0.,0.5,1.]:
            fig+=a_level(rel_h)
        for x in range(nr_x):
            fig+=a_bar(x,a[x])
        if entr_bool:
            fig+=text("Graph Entropy: {:.6f}".format(ge.phi_a(ge.a)),(-0.5,h0+th+1),horizontal_alignment='left',color='black',fontsize='large',fontweight='bold')    
    
    
    if grid_bool:
        fig+=grid(nr_x,nr_j)
        js=[*range(nr_j)]
        js.sort(key=lambda j:-r[j])
        for j_ind in range(nr_j):
            j=js[j_ind]
            s=sets[j]
            for x in range(nr_x):
                if s[x]>0:
                    fig+=ci(x,j_ind)    
            fig+=text(sets_lbl[j],(-1,-j_ind),horizontal_alignment='right',color='black')
            fig+=text("{:.4f}".format(r[j]),(nr_x+0.5,-j_ind),horizontal_alignment='left',color='blue')

    #print("Graph entropy: {}".format(ge.phi_a(ge.a)))
    #for x in range(nr_x):
    #    print("{}: {:.4f}".format(lbl[x],a[x]))
  
    fig.axes(False)
    if show_bool:
        fig.show(figsize=0.25*(nr_x+10),aspect_ratio=1)
    else:
        return fig

def plot_precision(errors,color='blue',step_size=1):
    ps=min(20,ceil(500./len(errors)))
    precs=[-log(err)/log(10.) for err in errors]
    steps=[*range(step_size,step_size*(1+len(errors)),step_size)]
    pts=list(zip(steps,precs))
    return list_plot(pts,color=color,pointsize=ps)

def error_tracker(ge,true_val,block=1,eps_stop=5e-15,pr_bool=True):
    if pr_bool:
        print("Value and errors after every {} iteration(s):".format(block))
    st=0
    vals=[]
    errs=[]
    ebs=[]
    for _ in range(ge.steps_max/block):
        st+=block
        ge.iter_step(block)
        val=ge.phi_a(ge.a)  #phi_r of previous r
        if val<=true_val+eps_stop:
            break
        eb=ge.opt_check()[0]
        vals.append(val)
        errs.append(val-true_val)
        ebs.append(eb)
        if ge.eps_active>0:
            ge.nullify()
        if pr_bool:
            print("#{:3d}: {:.16f}  error: {:.1e}  error bound: {:.1e}".format(st,val,val-true_val,eb))
    
    print()
    print("Plotting precision of iterative algorithm in terms of number of steps:")
    print("precision= nr of precise decimal digits= -log_10(error)")
    print("red points:  true precision")
    print("blue points: precision guaranteed by error bound based on the dual problem")
    fig=plot_precision(ebs,'blue',block)+plot_precision(errs,'red',block)
    fig.show(figsize=[8,4],axes_labels=['steps','precision'],axes=False,frame=True)
