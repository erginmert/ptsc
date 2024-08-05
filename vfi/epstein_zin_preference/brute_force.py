import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d
from scipy.optimize import brute


d = 0.9 # discount parameter of SC
ß = 0.2
ßd = ß*d # discount parameter of NSC
R = 1.05 # gross interest rate
k = 0.01 # cost of self-control
g = 0.8

w_max = 7
nk = 1000
ns = 1000
x_max = w_max # for graph

def maximize(w, xi, d, R, v, w_grid,ik,ns):
    '''
    This method maximized the sequantial problem.
'''

    p = 1 - 1 / g

    v_interp = interp1d(w_grid, v, kind='linear', fill_value="extrapolate")

    # Define objective function
    def f(c):
        return - np.power(((1-d) * np.power(c, p) + xi * np.power(v_interp(R * (w - c)) , p)), (1/p))

    # Define the search interval (bounds)
    search_interval = [(0, w)]

    # Run optimization using brute force
    res = brute(f, search_interval, full_output=True, finish=None, Ns=ns)

    # find optimal by dropping nan values
    cons = res[2]
    T = res[3]
    consOptInd= np.nanargmin(T)

    # Get the optimal solution and its objective function value
    sol = [cons[consOptInd], -T[consOptInd]]

    return sol

def VFIterator(d, ßd, R, k, g, w_max, nk):
    # help variable for coloring graphs
    color = 20

    time_start = time.time()
    print("Start time: " + str(time_start))

    p = 1 - 1 / g
    w_grid = np.linspace(0, w_max, nk)

    # value functions for SC and NSC (at the end v will be max{v_SC,v_NSC})
    v_SC = []
    v_SC.append([])

    v_NSC = []
    v_NSC.append([])
    # value function
    v = []
    v.append([])

    # initial guess
    #initial_guess = w_grid

    initial_guess = []
    w_bar = 3.18183189
    b_nsc = 0.022234297
    b_sc = 0.070274906
    a = R/(R-1) * k
    for w in w_grid:
        if w < w_bar:
            initial_guess.append(b_nsc*w)
        else:
            initial_guess.append(-a + b_sc*w)

    v[0] = initial_guess

    # derivatives
    v_SC_prime = []
    v_SC_prime.append([])

    v_NSC_prime = []
    v_NSC_prime.append([])

    v_prime = []
    v_prime.append([])

    # w'
    w_prime = []
    w_prime.append([])

    w_prime_SC = []
    w_prime_SC.append([])

    w_prime_NSC = []
    w_prime_NSC.append([])

    #capture whether SC exerted
    NSC_SC = []
    NSC_SC.append([])

    v_d_NSC = []
    v_d_NSC.append([])

    #fig, ax = plt.subplots()
    #ax.plot(w_grid, v[0], color=plt.cm.jet(0), lw=2, alpha=0.6, label='Initial guess')
    last = 0
    interrupted = ""
    c = True
    i = 1
    try:
        while(c):
            v_SC.append([])
            v_NSC.append([])
            v.append([])
            v_SC_prime.append([])
            v_NSC_prime.append([])
            v_prime.append([])
            w_prime.append([])
            w_prime_SC.append([])
            w_prime_NSC.append([])

            NSC_SC.append([])
            v_d_NSC.append([])
            # finding optimal wealth for tomorrow for all possible wealth levels in the grid
            # fix wealth
            for ik in range(0, nk):

                # v_SC
                sol_SC = maximize(w_grid[ik],d,d,R,v[i-1],w_grid,ik,ns)
                v_SC[i].append(sol_SC[1] - k)
                w_prime_SC[i].append(R * (w_grid[ik] - sol_SC[0]))
                if ik == 0:
                    v_SC_prime[i].append("")
                else:
                    v_SC_prime[i].append((v_SC[i][ik]-v_SC[i][ik-1])/(w_grid[ik]-w_grid[ik-1]))

                # v_NSC
                sol_NSC = maximize(w_grid[ik],ßd,d,R,v[i-1],w_grid,ik,ns)
                v_interp = interp1d(w_grid, v[i - 1], kind='linear', fill_value="extrapolate")
                v_NSC[i].append(((1-d) * (sol_NSC[0]) ** p + d * v_interp(R * (w_grid[ik] - sol_NSC[0])) ** p) ** (1/p))
                v_d_NSC[i].append(sol_NSC[1])
                w_prime_NSC[i].append(R * (w_grid[ik] - sol_NSC[0]))
                if ik == 0:
                    v_NSC_prime[i].append("")
                else:
                    v_NSC_prime[i].append((v_NSC[i][ik]-v_NSC[i][ik-1])/(w_grid[ik]-w_grid[ik-1]))

            # determine the value function for period T-t
            for ik in range(0, nk):
                if v_SC[i][ik] > v_NSC[i][ik]:
                    v[i].append(v_SC[i][ik])  # set value function
                    w_prime[i].append(w_prime_SC[i][ik])
                    NSC_SC[i].append(1)  # SC
                else:
                    v[i].append(v_NSC[i][ik])  # set value function
                    w_prime[i].append(w_prime_NSC[i][ik])
                    NSC_SC[i].append(0)  # NSC
                if ik == 0:
                    v_prime[i].append("")
                else:
                    v_prime[i].append((v[i][ik]-v[i][ik-1])/(w_grid[ik]-w_grid[ik-1]))

            # Graph #######
            fig, ax = plt.subplots()
            ax.set_title("V")
            ax.plot(w_grid, v_NSC[i], color="r")
            ax.plot(w_grid, v_SC[i], color="b")
            ax.set_xlim(0, x_max)
            y_max = v_SC[i][np.abs(w_grid - x_max).argmin()]
            y_min = v_SC[i][0]
            ax.set_ylim(y_min - 0.1, y_max)
            ax.tick_params(labelsize=14)
            ax.annotate(
                "d=" + str(round(d, 2)) + ", ß=" + str(round(ß, 2)) + ", R=" + str(round(R, 3)) + ", k=" + str(
                    k) + ", γ=" + str(g) + ", ß*R=" + str(
                    str(round(d * R, 2))) + " nk=" + str(nk) + " ns=" + str(ns) + ", Iter.=" + str(i),
                xy=(1.0, -0.2), xycoords='axes fraction', ha='right', va="center", fontsize=10)

            fig.tight_layout()
            plt.legend()
            plt.grid()
            plt.show()
            ##########
            # stop
            error = []
            for j in range(0,nk):
                error.append(v[i - 1][j] - v[i][j])
            error = np.max(np.abs(error))
            print("iteration = " + str(i) + ", error = " + str(error))
            if error < 0.000001:
                last = i
                c = False
            i = i + 1
    except KeyboardInterrupt:
        print("INTERRUPTED")
        interrupted = "_interrupted_iter_"+ str(len(v_SC)-2)

    time_end = time.time()
    print('Computational time: ', time_end-time_start, 'seconds')

    if last == 0:
        last = len(v_SC)-2

    fig, ax = plt.subplots()
    ax.set_title("V")
    ax.plot(w_grid, v_NSC[last], color="r")
    ax.plot(w_grid, v_SC[last], color="b")
    ax.set_xlim(0, x_max)
    y_max = v_SC[last][np.abs(w_grid - x_max).argmin()]
    y_min = v_SC[last][5]
    ax.set_ylim(y_min-0.1,y_max)
    ax.tick_params(labelsize=14)
    ax.annotate(
        "d=" + str(round(d,2)) + ", ß=" + str(round(ß,3)) + ", R=" + str(round(R,3)) + ", k=" + str(k) + ", γ=" + str(g) + ", d*R=" + str(
        str(round(d * R, 2))) + " nk=" + str(nk) + " ns=" + str(ns) +", Iter.=" + str(last),
        xy=(1.0, -0.2), xycoords='axes fraction', ha='right', va="center", fontsize=10)

    fig.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("W'")
    ax.plot(w_grid, w_prime_NSC[last], color="r")
    ax.plot(w_grid, w_prime_SC[last], color="b")
    ax.set_xlim(0, x_max)
    y_max = w_prime_SC[last][np.abs(w_grid - x_max).argmin()]
    y_min = w_prime_SC[last][5]
    ax.set_ylim(y_min - 0.1, y_max)
    ax.tick_params(labelsize=14)
    ax.annotate(
        "d=" + str(round(d, 2)) + ", ß=" + str(round(ß, 3)) + ", R=" + str(round(R, 3)) + ", k=" + str(
            k) + ", γ=" + str(g) + ", d*R=" + str(
            str(round(d * R, 2))) + " nk=" + str(nk) + " ns=" + str(ns) + ", Iter.=" + str(last),
        xy=(1.0, -0.2), xycoords='axes fraction', ha='right', va="center", fontsize=10)

    fig.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()


    # return the solution
    sol = {"w" : w_grid, "v" : v, "v_prime" : v_prime, "w_prime" : w_prime,
           "v_SC" : v_SC, "dv_SC_prime" : v_SC_prime, "w_prime_SC" : w_prime_SC,
           "v_NSC" : v_NSC, "dv_NSC_prime" : v_NSC_prime, "w_prime_NSC" : w_prime_NSC}
    # Excel
    df = pd.DataFrame(list(
        zip(w_grid, v[last], v_prime[last], w_prime[last], v_SC[last], v_SC_prime[last], w_prime_SC[last],
            v_NSC[last], v_NSC_prime[last], w_prime_NSC[last], v_d_NSC[last])),
        columns=["W", "V", "dv/dw", "W_prime", "v_SC", "dv_SC/dw", "W_prime_SC",
                 "v_NSC", "dv_NSC/dw", "W_prime_NSC","v_d_NSC"])
    dataToExcel = pd.ExcelWriter('d_'+ str(d) + "_ß_" + str(ß) + "_R_" + str(R)+"_k_"+ str(k) + "_g_"+ str(g) + interrupted + ".xlsx")
    df.to_excel(dataToExcel)
    dataToExcel.save()
    return sol

sol = VFIterator(d,ßd,R,k,g,w_max,nk)

'''
ß_grid = np.linspace(0.2,1,21)
for i in range(1,19):
    VFIterator(ß_grid[i],d,1.001,k,g,w,nk)
'''