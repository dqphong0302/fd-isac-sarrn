import pandas as pd, numpy as np, math, sys
F=sys.argv[1]; S=sys.argv[2]
T=pd.read_csv(F+'nmse_ber_table.csv')
def r(n,m,g): return T[(T.np==n)&(T.method==m)&(T.gamma_db==g)].iloc[0]
def sci(x):
    if x==0: return "0"
    e=math.floor(math.log10(x)); return f"${x/10**e:.2f}\\times10^{{{e}}}$"
def head(cap,lab,cols,hdr,resize=True):
    return ["\\begin{table}[!ht]","\\centering",f"\\caption{{{cap}}}",f"\\label{{{lab}}}"]+(["\\resizebox{\\linewidth}{!}{%"] if resize else [])+[f"\\begin{{tabular}}{{{cols}}}","\\toprule",hdr+" \\\\ \\midrule"]
def tail(resize=True): return ["\\bottomrule","\\end{tabular}"+("}" if resize else ""),"\\end{table}"]
G=[0,4,10,14,20,30]
rows=[(1,'ls','LS, 1 pilot'),(2,'ls','LS, 2 pilots'),(2,'rls','RLS (reset per frame), 2 pilots'),(2,'rls_carryP','RLS as submitted ($\\mathbf{P}$ not reset), 2 pilots'),(1,'lmmse','LMMSE (true PDP), 1 pilot'),(1,'ridge','Ridge anchor, no network, 1 pilot'),(1,'sarrn_v36','Network as submitted, 1 pilot'),(1,'sarrn_etilde','Network retrained, pilot anchor, 1 pilot'),(1,'ddls','DD-LS, no network, 1 pilot'),(1,'sarrn_da','\\textbf{SARRN (proposed), 1 pilot}')]
L=head("{\\bf NMSE (dB) of all estimators on the same test frames.} Mean over 50 frames per point, final IFD-DIC iteration. The 95\\,\\% bootstrap confidence intervals (half-width at most 0.9\\,dB) are shown in Fig~\\ref{fig3} and listed in S1 File.","tab4","l"+"c"*len(G),"Estimator & "+" & ".join(f"$\\gamma_c={g}$\\,dB" for g in G))
for n,m,lab in rows:
    L.append(lab+" & "+" & ".join(f"${x.nmse_db:.1f}$" for x in [r(n,m,g) for g in G])+" \\\\")
open(S+'/tab4.tex','w').write("\n".join(L+tail()))
G2=[0,6,10,14,20]
rows2=[(1,'ls','LS, 1 pilot'),(2,'ls','LS, 2 pilots'),(2,'rls','RLS (reset), 2 pilots'),(1,'ddls','DD-LS, 1 pilot'),(1,'sarrn_da','SARRN, 1 pilot'),(1,'genie','Perfect CSI')]
L=head("{\\bf BER of the estimators and with perfect CSI.} Errors pooled over 50 frames ($26.2\\times10^6$ bits per point for $N_p=1$); when no error occurred the one-sided 95\\,\\% upper bound is given. The 95\\,\\% Clopper--Pearson intervals (relative half-width below 6\\,\\% for $\\gamma_c\\le14$\\,dB) are listed in S1 File.","tab6","l"+"c"*len(G2),"Estimator & "+" & ".join(f"$\\gamma_c={g}$\\,dB" for g in G2))
for n,m,lab in rows2:
    c=[]
    for g in G2:
        x=r(n,m,g)
        c.append(sci(x.ber) if x.bit_errors>0 else f"0 ($<${sci(x.ber_hi)})")
    L.append(lab+" & "+" & ".join(c)+" \\\\")
open(S+'/tab6.tex','w').write("\n".join(L+tail()))
R=pd.read_csv(F+'radar_summary.csv')
L=head("{\\bf Sensing performance over 300 random three-target scenes per $\\gamma_c$.} Association gate: one native resolution cell; receiver with the one-pilot DD-LS channel estimate. ``Submitted'': original periodogram mapping, for reference.","tab5","lcccccc","Periodogram & $\\gamma_c$ (dB) & Iter. & $P_d$ & False det./frame & RMSE $R$ (m) & RMSE $v$ (m/s)")
for _,x in R.iterrows():
    L.append(f"{'corrected' if x['mode']=='fixed' else 'submitted'} & {x.gamma_db:.0f} & {x.fb} & {x.Pd:.3f} & {x.false_per_frame:.2f} & {x.rmse_r_m:.3f} & {x.rmse_v_mps:.3f}\\\\")
open(S+"/tab5.tex","w").write("\n".join(L+tail()))
M=pd.read_csv(F+'mismatch_frames.csv')
A=M.groupby(['variant','method','gamma_db']).agg(nmse=('nmse','mean'),ber=('ber','mean')).reset_index()
V=['nominal','doppler_100','doppler_500','doppler_1000','taps6','pdp_flat','si_fast']
VN={'nominal':'Nominal','doppler_100':'Doppler 100 Hz','doppler_500':'Doppler 500 Hz','doppler_1000':'Doppler 1 kHz','taps6':'6 taps (2 unmodelled)','pdp_flat':'PDP $[0,-3,-6,-9]$ dB','si_fast':'SI reflections 3 m/s'}
MN={'ls_1p':'LS-1','ls_2p':'LS-2','ddls_1p':'DD-LS','sarrn_da_1p':'SARRN'}
L=head("{\\bf NMSE (dB) / BER under model mismatch, without retraining.} 20 frames per entry. LS-1/LS-2: LS with one/two pilots; DD-LS and SARRN use one pilot.","tab8","llccc","Variant & Estimator & $\\gamma_c=10$\\,dB & $\\gamma_c=20$\\,dB & $\\gamma_c=30$\\,dB")
for v in V:
    for j,m in enumerate(MN):
        c=[]
        for g in (10,20,30):
            x=A[(A.variant==v)&(A.method==m)&(A.gamma_db==g)].iloc[0]
            c.append(f"${10*np.log10(x.nmse):.1f}$ / {sci(x.ber)}")
        L.append((VN[v] if j==0 else "")+f" & {MN[m]} & "+" & ".join(c)+"\\\\")
    if v!=V[-1]: L.append("\\midrule")
open(S+'/tab8.tex','w').write("\n".join(L+tail()))
print(A[A.variant.isin(['nominal','si_fast'])].assign(db=lambda d:10*np.log10(d.nmse)).to_string())
