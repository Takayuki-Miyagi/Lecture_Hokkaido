#include "Profiler.hh"
#include "Commutator.hh"
#include "omp_complex_reduction.hh"

namespace Commutator{

  Operator Commutator(const Operator& X, const Operator& Y)
  {
    if((not X.IsHamLikeSymmetry()) and (not Y.IsHamLikeSymmetry())){
      std::cout << "You are trying some complicated commutator, but it is not properly implemented!" << std::endl;
      exit(0);
    }
    Profiler::counter["N_Commutators"]++;
    Operator Z = Operator();
    if(not X.IsHamLikeSymmetry()) Z.Allocate(*X.modelspace, X.rankTz, X.Qx, X.Qy, X.Qz);
    if(not Y.IsHamLikeSymmetry()) Z.Allocate(*X.modelspace, Y.rankTz, Y.Qx, Y.Qy, Y.Qz);
    if(X.IsHamLikeSymmetry() and Y.IsHamLikeSymmetry()) Z.Allocate(*X.modelspace, 0, 0, 0, 0);

    if((not X.antisym) and (not Y.antisym)) Z.SetAntiSym(true);
    if(X.antisym and Y.antisym) Z.SetAntiSym(true);
    comm110(X, Y, Z);
    comm220(X, Y, Z);
    comm111(X, Y, Z);
    comm121(X, Y, Z);
    comm221(X, Y, Z);
    comm122(X, Y, Z);
    comm222(X, Y, Z);
    return Z;
  }

  void comm110(const Operator& X, const Operator& Y, Operator& Z)
  {
    // I do nothing here as the one-body part of the generator is 0
  }

  void comm220(const Operator& X, const Operator& Y, Operator& Z)
  {
    if(Z.antisym) return;
    std::complex<double> res = {0.0, 0.0};
    if(use_naive_implementation){
      double start_time = omp_get_wtime();
      for(int i : Z.modelspace->hole_states){
        for(int j : Z.modelspace->hole_states){
          for(int a : Z.modelspace->particle_states){
            for(int b : Z.modelspace->particle_states){
              res += 0.5 * X.Get2BME(i,j,a,b) * Y.Get2BME(a,b,i,j);
            }
          }
        }
      }
      Z.ZeroBody += res;
      Profiler::timer[__func__] += omp_get_wtime() - start_time;
      return;
    }

    double start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic) reduction(+: res)
    for(int ich=0; ich<Z.modelspace->TwoBody.GetNumberChannels(); ich++){
      TwoBodyChannel& tbc = Z.modelspace->TwoBody.GetChannel(ich);
      for(int idx_hh : tbc.index_hh){
        int ii = tbc.GetSPStateIndex1(idx_hh);
        int ij = tbc.GetSPStateIndex2(idx_hh);
        SPState& i = Z.modelspace->GetSPState(ii);
        SPState& j = Z.modelspace->GetSPState(ij);
        for(int idx_pp : tbc.index_pp){
          int ia = tbc.GetSPStateIndex1(idx_pp);
          int ib = tbc.GetSPStateIndex2(idx_pp);
          SPState& a = Z.modelspace->GetSPState(ia);
          SPState& b = Z.modelspace->GetSPState(ib);
          res += 2.0 * i.occ * j.occ * (1.0-a.occ) * (1.0-b.occ) * (X.Get2BME(ii,ij,ia,ib) * Y.Get2BME(ia,ib,ii,ij));
        }
      }
    }
    Z.ZeroBody += res;
    Profiler::timer[__func__] += omp_get_wtime() - start_time;
  }

  void comm111(const Operator& X, const Operator& Y, Operator& Z)
  {
    // I do nothing here as the one-body part of the generator is 0
  }

  void comm121(const Operator& X, const Operator& Y, Operator& Z)
  {
    double start_time = omp_get_wtime();
//#pragma omp parallel for schedule(static)
#pragma omp parallel for schedule(dynamic)
    for(int ip : Z.modelspace->all_states){
      for(int iq : Z.OneBody.non_zero_loop_indices.at(ip)){

        std::complex<double> me = {0.0, 0.0};
        for(int ia : Z.modelspace->all_states){
          SPState& a = Z.modelspace->GetSPState(ia);
          for(int ib : Z.modelspace->all_states){
            SPState& b = Z.modelspace->GetSPState(ib);
            if(std::abs(a.occ - b.occ) < 1.e-4) continue;
            me += (a.occ - b.occ) * (X.Get1BME(ia,ib) * Y.Get2BME(ib,ip,ia,iq) - Y.Get1BME(ia,ib) * X.Get2BME(ib,ip,ia,iq));
          }
        }
        Z.OneBody.Matrix(ip,iq) += me;
      }
    }
    Profiler::timer[__func__] += omp_get_wtime() - start_time;
  }

  void comm221(const Operator& X, const Operator& Y, Operator& Z)
  {
    // naive implementation
    if(use_naive_implementation){
      double start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
      for(int ip : Z.modelspace->all_states){
        for(int iq : Z.OneBody.non_zero_loop_indices.at(ip)){

          std::complex<double> me = {0.0, 0.0};
          for(int ia : Z.modelspace->all_states){
            SPState& a = Z.modelspace->GetSPState(ia);
            for(int ib : Z.modelspace->all_states){
              SPState& b = Z.modelspace->GetSPState(ib);
              for(int ic : Z.modelspace->all_states){
                SPState& c = Z.modelspace->GetSPState(ic);
                me += 0.5 * ((1-a.occ)*(1-b.occ)*c.occ + a.occ*b.occ*(1-c.occ)) *
                  (X.Get2BME(ip,ic,ia,ib)*Y.Get2BME(ia,ib,iq,ic) - Y.Get2BME(ip,ic,ia,ib)*X.Get2BME(ia,ib,iq,ic));
              }
            }
          }
          Z.OneBody.Matrix(ip, iq) += me;
        }
      }
      Profiler::timer[__func__] += omp_get_wtime() - start_time;
      return;
    }

    double start_time = omp_get_wtime();
    Operator Zop_occ = Z;
    Operator Zop_unocc = Z;
    std::vector<std::array<int,3>> looplist;
    for(auto& it_X : X.TwoBody.Matrices){
      int xichbra = it_X.first[0];
      int xichket = it_X.first[1];
      for(auto& it_Y : Y.TwoBody.Matrices){
        int yichbra = it_Y.first[0];
        int yichket = it_Y.first[1];
        if(xichket != yichbra) continue;
        looplist.push_back({xichbra,xichket,yichket});
      }
    }

#pragma omp parallel for schedule(dynamic)
    for(int loop_index=0; loop_index<looplist.size(); loop_index++){
      int ichbra = looplist[loop_index][0];
      int ichmid = looplist[loop_index][1];
      int ichket = looplist[loop_index][2];

      auto& Xop = X.TwoBody.Matrices.at({ichbra,ichmid});
      auto& Yop = Y.TwoBody.Matrices.at({ichmid,ichket});

      TwoBodyChannel& chmid = Z.modelspace->TwoBody.GetChannel(ichmid);
      TwoBodyChannel& chbra = Z.modelspace->TwoBody.GetChannel(ichbra);
      TwoBodyChannel& chket = Z.modelspace->TwoBody.GetChannel(ichket);

      if(chmid.index_hh.size() > 0){
        Eigen::MatrixXcd xsub(chbra.GetNumberStates(), chmid.index_hh.size());
        Eigen::MatrixXcd ysub(chmid.index_hh.size(), chket.GetNumberStates());
        for (size_t idx=0; idx<chmid.index_hh.size(); ++idx) {
          int ii = chmid.GetSPStateIndex1(chmid.index_hh[idx]);
          int ij = chmid.GetSPStateIndex2(chmid.index_hh[idx]);
          SPState& i = Z.modelspace->GetSPState(ii);
          SPState& j = Z.modelspace->GetSPState(ij);
          xsub.col(idx) = Xop.col(chmid.index_hh[idx]) * i.occ * j.occ;
          ysub.row(idx) = Yop.row(chmid.index_hh[idx]);
        }
        Zop_occ.TwoBody.Matrices.at({ichbra,ichket}) = xsub * ysub;
      }

      if(chmid.index_pp.size() > 0){
        Eigen::MatrixXcd xsub(chbra.GetNumberStates(), chmid.index_pp.size());
        Eigen::MatrixXcd ysub(chmid.index_pp.size(), chket.GetNumberStates());
        for (size_t idx=0; idx<chmid.index_pp.size(); ++idx) {
          int ia = chmid.GetSPStateIndex1(chmid.index_pp[idx]);
          int ib = chmid.GetSPStateIndex2(chmid.index_pp[idx]);
          SPState& a = Z.modelspace->GetSPState(ia);
          SPState& b = Z.modelspace->GetSPState(ib);
          xsub.col(idx) = Xop.col(chmid.index_pp[idx]) * (1.0 - a.occ) * (1.0 - b.occ);
          ysub.row(idx) = Yop.row(chmid.index_pp[idx]);
        }
        Zop_unocc.TwoBody.Matrices.at({ichbra,ichket}) = xsub * ysub;
      }
    }

    looplist.clear();
    for(auto& it_Y : Y.TwoBody.Matrices){
      int yichbra = it_Y.first[0];
      int yichket = it_Y.first[1];
      for(auto& it_X : X.TwoBody.Matrices){
        int xichbra = it_X.first[0];
        int xichket = it_X.first[1];
        if(yichket != xichbra) continue;
        looplist.push_back({yichbra,yichket,xichket});
      }
    }

#pragma omp parallel for schedule(dynamic)
    for(int loop_index=0; loop_index<looplist.size(); loop_index++){
      int ichbra = looplist[loop_index][0];
      int ichmid = looplist[loop_index][1];
      int ichket = looplist[loop_index][2];

      auto& Yop = Y.TwoBody.Matrices.at({ichbra,ichmid});
      auto& Xop = X.TwoBody.Matrices.at({ichmid,ichket});

      TwoBodyChannel& chmid = Z.modelspace->TwoBody.GetChannel(ichmid);
      TwoBodyChannel& chbra = Z.modelspace->TwoBody.GetChannel(ichbra);
      TwoBodyChannel& chket = Z.modelspace->TwoBody.GetChannel(ichket);

      Eigen::MatrixXcd ysub(chbra.GetNumberStates(), chmid.index_hh.size());
      Eigen::MatrixXcd xsub(chmid.index_hh.size(), chket.GetNumberStates());
      for (size_t idx=0; idx<chmid.index_hh.size(); ++idx) {
        int ii = chmid.GetSPStateIndex1(chmid.index_hh[idx]);
        int ij = chmid.GetSPStateIndex2(chmid.index_hh[idx]);
        SPState& i = Z.modelspace->GetSPState(ii);
        SPState& j = Z.modelspace->GetSPState(ij);
        ysub.col(idx) = Yop.col(chmid.index_hh[idx]) * i.occ * j.occ;
        xsub.row(idx) = Xop.row(chmid.index_hh[idx]);
      }
      Zop_occ.TwoBody.Matrices.at({ichbra,ichket}) -= ysub * xsub;

      ysub.resize(chbra.GetNumberStates(), chmid.index_pp.size());
      xsub.resize(chmid.index_pp.size(), chket.GetNumberStates());
      for (size_t idx=0; idx<chmid.index_pp.size(); ++idx) {
        int ia = chmid.GetSPStateIndex1(chmid.index_pp[idx]);
        int ib = chmid.GetSPStateIndex2(chmid.index_pp[idx]);
        SPState& a = Z.modelspace->GetSPState(ia);
        SPState& b = Z.modelspace->GetSPState(ib);
        ysub.col(idx) = Yop.col(chmid.index_pp[idx]) * (1.0 - a.occ) * (1.0 - b.occ);
        xsub.row(idx) = Xop.row(chmid.index_pp[idx]);
      }
      Zop_unocc.TwoBody.Matrices.at({ichbra,ichket}) -= ysub * xsub;
    }

    std::vector<std::array<int,2>> tmp;
    for(int ip : Z.modelspace->all_states){
      for(int iq : Z.OneBody.non_zero_loop_indices.at(ip)){
        tmp.push_back({ip,iq});
      }
    }

#pragma omp parallel for schedule(dynamic)
//#pragma omp parallel for schedule(static)
    for(int it=0; it<tmp.size(); it++){
      int ip = tmp[it][0];
      int iq = tmp[it][1];
      std::complex<double> me = {0.0, 0.0};
      for(int ic : Z.modelspace->all_states){
        SPState& c = Z.modelspace->GetSPState(ic);
        me += Zop_occ.Get2BME(ip,ic,iq,ic) * (1.0-c.occ);
        me += Zop_unocc.Get2BME(ip,ic,iq,ic) * c.occ;
      }
      Z.OneBody.Matrix(ip,iq) += me;
    }
    Profiler::timer[__func__] += omp_get_wtime() - start_time;
  }

  void comm122(const Operator& X, const Operator& Y, Operator& Z)
  {

    bool use_x1_0 = false;
    bool use_y1_0 = false;
    bool use_x1_0_y1_diag = false;
    bool use_x1_diag_y1_0 = false;
    if(X.OneBody.IsZero(1.e-8)){
      if(Y.OneBody.IsDiagonal(1.e-8)) use_x1_0_y1_diag = true;
      else use_x1_0 = true;
    }
    if(Y.OneBody.IsZero(1.e-8)){
      if(X.OneBody.IsDiagonal(1.e-8)) use_x1_diag_y1_0 = true;
      else use_y1_0 = true;
    }

    // This is general, but slow...
    auto me_func = [&X, &Y, use_x1_0, use_y1_0, use_x1_0_y1_diag, use_x1_diag_y1_0](int ip, int iq, int ir, int is){
      std::complex<double> me = {0.0, 0.0};
      if(use_x1_0){
        for(int ia : Y.OneBody.non_zero_loop_indices.at(ip)) me -= Y.Get1BME(ip,ia) * X.Get2BME(ia,iq,ir,is);
        for(int ia : Y.OneBody.non_zero_loop_indices.at(iq)) me += Y.Get1BME(iq,ia) * X.Get2BME(ia,ip,ir,is);
        for(int ia : Y.OneBody.non_zero_loop_indices.at(ir)) me += Y.Get1BME(ia,ir) * X.Get2BME(ip,iq,ia,is);
        for(int ia : Y.OneBody.non_zero_loop_indices.at(is)) me -= Y.Get1BME(ia,is) * X.Get2BME(ip,iq,ia,ir);
      }
      else if(use_y1_0){
        for(int ia : X.OneBody.non_zero_loop_indices.at(ip)) me += X.Get1BME(ip,ia) * Y.Get2BME(ia,iq,ir,is);
        for(int ia : X.OneBody.non_zero_loop_indices.at(iq)) me -= X.Get1BME(iq,ia) * Y.Get2BME(ia,ip,ir,is);
        for(int ia : X.OneBody.non_zero_loop_indices.at(ir)) me -= X.Get1BME(ia,ir) * Y.Get2BME(ip,iq,ia,is);
        for(int ia : X.OneBody.non_zero_loop_indices.at(is)) me += X.Get1BME(ia,is) * Y.Get2BME(ip,iq,ia,ir);
      }
      else if(use_x1_0_y1_diag){
        std::complex<double> xme = X.Get2BME(ip,iq,ir,is);
        me -= Y.Get1BME(ip,ip) * xme; // pqrs
        me -= Y.Get1BME(iq,iq) * xme; // qprs
        me += Y.Get1BME(ir,ir) * xme; // pqrs
        me += Y.Get1BME(is,is) * xme; // pqsr
      }
      else if(use_x1_diag_y1_0){
        std::complex<double> yme = Y.Get2BME(ip,iq,ir,is);
        me += X.Get1BME(ip,ip) * yme; // pqrs
        me += X.Get1BME(iq,iq) * yme; // qprs
        me -= X.Get1BME(ir,ir) * yme; // pqrs
        me -= X.Get1BME(is,is) * yme; // pqsr
      }
      else{
        for(int ia : X.OneBody.non_zero_loop_indices.at(ip)) me += X.Get1BME(ip,ia) * Y.Get2BME(ia,iq,ir,is);
        for(int ia : X.OneBody.non_zero_loop_indices.at(iq)) me -= X.Get1BME(iq,ia) * Y.Get2BME(ia,ip,ir,is);
        for(int ia : X.OneBody.non_zero_loop_indices.at(ir)) me -= X.Get1BME(ia,ir) * Y.Get2BME(ip,iq,ia,is);
        for(int ia : X.OneBody.non_zero_loop_indices.at(is)) me += X.Get1BME(ia,is) * Y.Get2BME(ip,iq,ia,ir);

        for(int ia : Y.OneBody.non_zero_loop_indices.at(ip)) me -= Y.Get1BME(ip,ia) * X.Get2BME(ia,iq,ir,is);
        for(int ia : Y.OneBody.non_zero_loop_indices.at(iq)) me += Y.Get1BME(iq,ia) * X.Get2BME(ia,ip,ir,is);
        for(int ia : Y.OneBody.non_zero_loop_indices.at(ir)) me += Y.Get1BME(ia,ir) * X.Get2BME(ip,iq,ia,is);
        for(int ia : Y.OneBody.non_zero_loop_indices.at(is)) me -= Y.Get1BME(ia,is) * X.Get2BME(ip,iq,ia,ir);
      }

      return me;
    };

    std::vector<std::array<int,2>> tmp;
    for(auto& it : Z.TwoBody.Matrices){
      tmp.push_back(it.first);
    }
    double start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
//#pragma omp parallel for schedule(static)
    for(int chindices=0; chindices<tmp.size(); chindices++){
      int ichbra = tmp[chindices][0];
      int ichket = tmp[chindices][1];
      TwoBodyChannel& tbcbra = Z.modelspace->TwoBody.GetChannel(ichbra);
      TwoBodyChannel& tbcket = Z.modelspace->TwoBody.GetChannel(ichket);
      for(int idx_bra=0; idx_bra<tbcbra.GetNumberStates(); idx_bra++){
        int ip = tbcbra.GetSPStateIndex1(idx_bra);
        int iq = tbcbra.GetSPStateIndex2(idx_bra);
        int idx_ket_min = 0;
        if(ichbra==ichket) idx_ket_min = idx_bra;

        for(int idx_ket=idx_ket_min; idx_ket<tbcket.GetNumberStates(); idx_ket++){
          int ir = tbcket.GetSPStateIndex1(idx_ket);
          int is = tbcket.GetSPStateIndex2(idx_ket);

          std::complex<double> me = me_func(ip, iq, ir, is);
          Z.TwoBody.Matrices.at({ichbra,ichket})(idx_bra,idx_ket) += me;

          if(ichbra!=ichket) continue;
          if(idx_bra==idx_ket) continue;

          if(Z.antisym) {
            Z.TwoBody.Matrices.at({ichbra,ichket})(idx_ket,idx_bra) += std::conj(me) * -1.0;
          }
          else {
            Z.TwoBody.Matrices.at({ichbra,ichket})(idx_ket,idx_bra) += std::conj(me);
          }
        }
      }
    }
    Profiler::timer[__func__] += omp_get_wtime() - start_time;
  }


  void comm222(const Operator& X, const Operator& Y, Operator& Z)
  {
    // naive implementation
    if(use_naive_implementation){
      // pp and hh diagrams
      double start_time = omp_get_wtime();
      for(auto& it : Z.TwoBody.Matrices){
        int ichbra = it.first[0];
        int ichket = it.first[1];

        TwoBodyChannel& tbcbra = Z.modelspace->TwoBody.GetChannel(ichbra);
        TwoBodyChannel& tbcket = Z.modelspace->TwoBody.GetChannel(ichket);
#pragma omp parallel for
        for(int idx_bra=0; idx_bra<tbcbra.GetNumberStates(); idx_bra++){
          int ip = tbcbra.GetSPStateIndex1(idx_bra);
          int iq = tbcbra.GetSPStateIndex2(idx_bra);
          for(int idx_ket=0; idx_ket<tbcket.GetNumberStates(); idx_ket++){
            int ir = tbcket.GetSPStateIndex1(idx_ket);
            int is = tbcket.GetSPStateIndex2(idx_ket);
            std::complex<double> me = {0.0, 0.0};
            for(int ia : Z.modelspace->all_states){
              SPState& a = Z.modelspace->GetSPState(ia);
              for(int ib : Z.modelspace->all_states){
                SPState& b = Z.modelspace->GetSPState(ib);
                me += 0.5 * ((1-a.occ)*(1-b.occ) - a.occ*b.occ) * (X.Get2BME(ip,iq,ia,ib)*Y.Get2BME(ia,ib,ir,is)- Y.Get2BME(ip,iq,ia,ib)*X.Get2BME(ia,ib,ir,is));
              }
            }
            Z.TwoBody.Matrices.at({ichbra,ichket})(idx_bra,idx_ket) += me;
          }
        }
      }
      Profiler::timer["comm222 (pp & hh)"] += omp_get_wtime() - start_time;

      // ph diagram
      start_time = omp_get_wtime();
      for(auto& it : Z.TwoBody.Matrices){
        int ichbra = it.first[0];
        int ichket = it.first[1];
        auto& Zmat = it.second;

        TwoBodyChannel& tbcbra = Z.modelspace->TwoBody.GetChannel(ichbra);
        TwoBodyChannel& tbcket = Z.modelspace->TwoBody.GetChannel(ichket);
#pragma omp parallel for schedule(dynamic)
//#pragma omp parallel for schedule(static)
        for(int idx_bra=0; idx_bra<tbcbra.GetNumberStates(); idx_bra++){
          int ip = tbcbra.GetSPStateIndex1(idx_bra);
          int iq = tbcbra.GetSPStateIndex2(idx_bra);
          for(int idx_ket=0; idx_ket<tbcket.GetNumberStates(); idx_ket++){
            int ir = tbcket.GetSPStateIndex1(idx_ket);
            int is = tbcket.GetSPStateIndex2(idx_ket);
            std::complex<double> me = {0.0, 0.0};
            for(int ia : Z.modelspace->all_states){
              SPState& a = Z.modelspace->GetSPState(ia);
              for(int ib : Z.modelspace->all_states){
                SPState& b = Z.modelspace->GetSPState(ib);
                if(std::abs(a.occ - b.occ) < 1.e-8) continue;
                me += (a.occ - b.occ) *
                  (X.Get2BME(ia,ip,ib,ir)*Y.Get2BME(ib,iq,ia,is) - X.Get2BME(ia,iq,ib,ir)*Y.Get2BME(ib,ip,ia,is)-
                   X.Get2BME(ia,ip,ib,is)*Y.Get2BME(ib,iq,ia,ir) + X.Get2BME(ia,iq,ib,is)*Y.Get2BME(ib,ip,ia,ir));
              }
            }
            Z.TwoBody.Matrices.at({ichbra,ichket})(idx_bra,idx_ket) += me;
          }
        }
      }
      Profiler::timer["comm222 ph"] += omp_get_wtime() - start_time;
      return;
    }

    std::vector<std::array<int,2>> looplist2;
    for(auto& it : Z.TwoBody.Matrices){
      looplist2.push_back(it.first);
    }

    double start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
//#pragma omp parallel for schedule(static)
    for(int chindices=0; chindices<looplist2.size(); chindices++){
      int ichbra = looplist2[chindices][0];
      int ichket = looplist2[chindices][1];
      TwoBodyChannel& tbcbra = Z.modelspace->TwoBody.GetChannel(ichbra);
      TwoBodyChannel& tbcket = Z.modelspace->TwoBody.GetChannel(ichket);
      auto& Zmat = Z.TwoBody.Matrices.at(looplist2[chindices]);

      for(int ichinter=0; ichinter<Z.modelspace->TwoBody.GetNumberChannels(); ichinter++){
        TwoBodyChannel& chinter = Z.modelspace->TwoBody.GetChannel(ichinter);

        if(X.TwoBody.Matrices.find({ichbra,ichinter}) != X.TwoBody.Matrices.end()
            and Y.TwoBody.Matrices.find({ichinter,ichket}) != Y.TwoBody.Matrices.end()){
          Eigen::MatrixXcd Xmat(tbcbra.GetNumberStates(), chinter.index_for_comm_pphh.size());
          Eigen::MatrixXcd Ymat(chinter.index_for_comm_pphh.size(), tbcket.GetNumberStates());
          for (size_t i=0; i<chinter.index_for_comm_pphh.size(); ++i) {
            Xmat.col(i) = X.TwoBody.Matrices.at({ichbra,ichinter}).col(chinter.index_for_comm_pphh[i]) * chinter.factor_for_comm_pphh[i];
            Ymat.row(i) = Y.TwoBody.Matrices.at({ichinter,ichket}).row(chinter.index_for_comm_pphh[i]);
          }
          Zmat += Xmat * Ymat;
        }

        if(Y.TwoBody.Matrices.find({ichbra,ichinter}) != Y.TwoBody.Matrices.end()
            and X.TwoBody.Matrices.find({ichinter,ichket}) != X.TwoBody.Matrices.end()){
          Eigen::MatrixXcd Ymat(tbcbra.GetNumberStates(), chinter.index_for_comm_pphh.size());
          Eigen::MatrixXcd Xmat(chinter.index_for_comm_pphh.size(), tbcket.GetNumberStates());
          for (size_t i=0; i<chinter.index_for_comm_pphh.size(); ++i) {
            Ymat.col(i) = Y.TwoBody.Matrices.at({ichbra,ichinter}).col(chinter.index_for_comm_pphh[i]) * chinter.factor_for_comm_pphh[i];
            Xmat.row(i) = X.TwoBody.Matrices.at({ichinter,ichket}).row(chinter.index_for_comm_pphh[i]);
          }
          Zmat -= Ymat * Xmat;
        }
      }
    }
    Profiler::timer["comm222 (pp & hh)"] += omp_get_wtime() - start_time;

    start_time = omp_get_wtime();

    X.TwoBody.EnsureXOperator();
    Y.TwoBody.EnsureXOperator();
    TwoBodyOperator tmp_Z(*Z.modelspace, Z.rankTz, Z.Qx, Z.Qy, Z.Qz);
    tmp_Z.SetAntiSym(Z.antisym);
    tmp_Z.AllocateXOperator();

    std::vector<std::array<int,3>> looplist3;
    for(auto& it_X : X.TwoBody.XMatrices){
      int xichbra = it_X.first[0];
      int xichket = it_X.first[1];
      for(auto& it_Y : Y.TwoBody.XMatrices){
        int yichbra = it_Y.first[0];
        int yichket = it_Y.first[1];
        if(xichket != yichbra) continue;
        looplist3.push_back({xichbra,xichket,yichket});
      }
    }

#pragma omp parallel for schedule(dynamic)
    for (int chindices=0; chindices<looplist3.size(); ++chindices) {
      int ichbra = looplist3[chindices][0];
      int ichmid = looplist3[chindices][1];
      int ichket = looplist3[chindices][2];

      XTwoBodyChannel& chbra = Z.modelspace->TwoBody.GetXChannel(ichbra);
      XTwoBodyChannel& chket = Z.modelspace->TwoBody.GetXChannel(ichket);
      XTwoBodyChannel& chmid = Z.modelspace->TwoBody.GetXChannel(ichmid);
      if (chmid.index_for_comm_ph.size() == 0) continue;

      const auto& xmat = X.TwoBody.XMatrices.at({ichbra,ichmid});
      const auto& ymat = Y.TwoBody.XMatrices.at({ichmid,ichket});
      auto&       zmat = tmp_Z.XMatrices.at({ichbra,ichket});

      Eigen::MatrixXcd xsub(chbra.GetNumberStates(), chmid.index_for_comm_ph.size());
      Eigen::MatrixXcd ysub(chmid.index_for_comm_ph.size(), chket.GetNumberStates());
      for (size_t i=0; i<chmid.index_for_comm_ph.size(); ++i) {
        xsub.col(i) = xmat.col(chmid.index_for_comm_ph[i]) * chmid.factor_for_comm_ph[i];
        ysub.row(i) = ymat.row(chmid.index_for_comm_ph[i]);
      }
      zmat = xsub * ysub * 4.0;
    }

    tmp_Z.UpdateFromXOperator();
    Z.TwoBody += tmp_Z;

    Profiler::timer["comm222 ph"] += omp_get_wtime() - start_time;

  }
}
